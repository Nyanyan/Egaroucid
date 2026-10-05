"""Retire completed participants and reuse their process memory budget."""
from __future__ import annotations

import time
import threading
from functools import wraps

from strength_engine import available_memory_mib
from strength_tournament import atomic_write_json


def budget_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class GameThreadBudget:
    """Paired admission and acknowledged, TT-preserving live thread changes."""

    def __init__(self, specs, total=70, base_threads=1, max_threads=64):
        if base_threads < 1 or total < 4 * base_threads or max_threads < base_threads:
            raise ValueError("thread budget must fit both color games at the base thread count")
        self.specs = specs
        self.total = total
        self.base_threads = base_threads
        self.max_threads = max_threads
        self.allocations = {}
        self.game_players = {}
        self._lock = threading.RLock()
        self._live_targets = {}
        self._admitting = False
        self._reclaim = False
        self.live_thread_changes = 0

    @property
    @budget_locked
    def allocated(self):
        return self.allocated_search_threads + self.allocated_policy_processes

    @property
    @budget_locked
    def allocated_search_threads(self):
        return sum(sum(threads.values()) for threads in self.allocations.values())

    @property
    @budget_locked
    def allocated_policy_processes(self):
        return sum(self.specs[player].alpha == 1.0
                   for players in self.game_players.values() for player in players)

    @property
    @budget_locked
    def allocated_player_processes(self):
        return 2 * len(self.game_players)

    @budget_locked
    def begin_admission(self):
        self._admitting = True
        self._reclaim = False

    @budget_locked
    def reserve_set(self, task):
        players = [i for i in (task.p0_idx, task.p1_idx) if self.specs[i].alpha != 1.0]
        # Every game needs BOTH participants, including policy/random wrappers.
        # Reserve the complete pair before starting either side. Participant
        # limits follow from these pairs and the global budget, not a fixed cap.
        required = 2 * (len(players) * self.base_threads + 2 - len(players))
        if required > self.total - self.allocated:
            # Ask only runtime-capable engines to return their extra threads
            # at their next move boundary. Reservations remain until ACK.
            reclaimable = sum(threads - self.base_threads for plan in self.allocations.values()
                              for player, threads in plan.items() if self.specs[player].name.startswith("edax_"))
            self._reclaim |= required <= self.total - self.allocated + reclaimable
            return False
        for color in (True, False):
            key = (task.task_id, color)
            if key in self.allocations:
                raise ValueError("game already has a thread allocation")
            self.allocations[key] = dict.fromkeys(players, self.base_threads)
            self.game_players[key] = ((task.p0_idx, task.p1_idx) if color else (task.p1_idx, task.p0_idx))
        return True

    @budget_locked
    def distribute(self, tasks):
        # Reserve the whole admission batch before distributing its remainder,
        # so the first game cannot take the entire budget during startup.
        new_keys = {(task.task_id, color) for task in tasks for color in (True, False)}
        entries = [(key, player)
                   for task in tasks for color in (True, False)
                   for key in [(task.task_id, color)] for player in self.allocations[key]]
        live = [(key, player) for key, plan in self.allocations.items() if key not in new_keys
                for player in plan if self.specs[player].name.startswith("edax_")]
        self._live_targets = {entry: self.base_threads for entry in live}
        eligible = entries if self._reclaim else entries + live
        targets = {entry: self.base_threads for entry in eligible}
        remaining = self.total - self.allocated
        if not self._reclaim:
            remaining += sum(self.allocations[key][player] - self.base_threads for key, player in eligible)
        while remaining > 0:
            candidates = [(threads, index) for index, (entry, threads) in enumerate(targets.items())
                          if threads < self._maximum(entry[1])]
            if not candidates:
                break
            _, index = min(candidates)
            targets[eligible[index]] += 1
            remaining -= 1
        for key, player in entries:
            plan = self.allocations[key]
            plan[player] += min(targets[(key, player)] - plan[player], self.total - self.allocated)
        for entry in live:
            self._live_targets[entry] = targets.get(entry, self.base_threads)
        self._admitting = False

    def _maximum(self, player):
        return min(self.max_threads, 64) if self.specs[player].name.startswith("edax_") else self.max_threads

    @budget_locked
    def resize_at_move_boundary(self, task, color, player, change_threads):
        key = (task.task_id, color)
        if self._admitting or key not in self.allocations:
            return False
        plan = self.allocations[key]
        old = plan[player]
        desired = self._live_targets.get((key, player), old)
        target = min(desired, old + self.total - self.allocated)
        if target == old:
            return False
        # Serialize resize acknowledgement with admission and other resizes.
        # In particular a shrinking engine's slots aren't available early.
        change_threads(target)
        plan[player] = target
        self.live_thread_changes += 1
        return True

    @budget_locked
    def for_game(self, task, color):
        return dict(self.allocations[(task.task_id, color)])

    @budget_locked
    def release_game(self, task, color):
        self.allocations.pop((task.task_id, color))
        self.game_players.pop((task.task_id, color))
        self._live_targets = {entry: target for entry, target in self._live_targets.items()
                              if entry[0] != (task.task_id, color)}

    @budget_locked
    def process_counts(self):
        counts = [[0, 0] for _ in self.specs]
        for black, white in self.game_players.values():
            counts[black][0] += 1
            counts[white][1] += 1
        return counts

    @budget_locked
    def player_threads(self, player):
        return [plan[player] for plan in self.allocations.values() if player in plan]


def finish_priorities(specs, enabled):
    priorities = []
    for spec in specs:
        priority = 0
        if enabled and spec.alpha is None:
            level = int(spec.name.rsplit("_l", 1)[1])
            if level <= 9:
                priority = 10 - level
        priorities.append(priority)
    return priorities


def process_memory_costs(specs, args):
    costs = []
    for spec in specs:
        if spec.name.startswith("edax_"):
            cost = args.estimated_edax_memory_mib
        elif spec.alpha == 1.0:
            cost = args.estimated_wrapper_memory_mib
        else:
            cost = args.estimated_engine_memory_mib
            if spec.alpha is not None and spec.alpha > 0:
                cost += args.estimated_wrapper_memory_mib
        costs.append(cost)
    return costs


class AdaptiveEnginePools:
    MEMORY_MARGIN_MIB = 2048.0

    def __init__(self, specs, pools, costs, priorities, *, extra_memory_mib,
                 startup_available_mib, minimum_available_mib,
                 fixed_memory_mib, max_match_sets, output_path,
                 adaptive_threads=False, base_threads=1, thread_budget=None,
                 thread_memory_mib=64.0):
        self.specs = specs
        self.pools = pools
        self.costs = costs
        self.priorities = priorities
        self.capacities = [spec.concurrent_match_set_capacity for spec in specs]
        self.retired = [False] * len(specs)
        self.adaptive_threads = adaptive_threads
        self.base_threads = base_threads
        self.thread_memory_mib = thread_memory_mib
        self.thread_budget = (thread_budget or GameThreadBudget(specs, base_threads=base_threads)
                              if adaptive_threads else None)
        self.fixed_memory_mib = fixed_memory_mib
        self.active_counts = [0] * len(specs)
        self.growth_blocked = False
        self.initial_memory_mib = self.reserved_memory_mib
        self.extra_memory_mib = extra_memory_mib
        self.minimum_available_mib = minimum_available_mib
        self.max_match_sets = max_match_sets
        # Legacy mode reserves every permitted slot. Adaptive mode instead
        # reserves retained engines and admitted games against available RAM.
        self.budget_mib = max(self.initial_memory_mib, min(
            self.initial_memory_mib + extra_memory_mib,
            startup_available_mib - minimum_available_mib - self.MEMORY_MARGIN_MIB,
        ))
        if self.adaptive_threads:
            # The global thread budget bounds active engines. A per-player
            # capacity is only a scheduling limit, not a memory reservation.
            self.budget_mib = max(0.0, startup_available_mib - minimum_available_mib - self.MEMORY_MARGIN_MIB)
        self.output_path = output_path
        self.last_write = 0.0

    @property
    def reserved_memory_mib(self):
        # Additional threads use one global budget. Idle engines retaining
        # additional threads still consume memory and are counted separately.
        extra = 0.0
        if self.thread_budget is not None:
            extra = self.thread_memory_mib * (self.thread_budget.total + sum(
                pool.search_thread_status()["idle_extra_search_threads"] for pool in self.pools))
        if self.adaptive_threads:
            return self.fixed_memory_mib + extra + self._process_memory()[0]
        return self.fixed_memory_mib + extra + sum(
            2 * capacity * self._process_cost(index)
            for index, capacity in enumerate(self.capacities)
        )

    def _process_cost(self, index):
        extra_threads = self.base_threads - 1 if self.adaptive_threads and self.specs[index].alpha != 1.0 else 0
        return self.costs[index] + extra_threads * self.thread_memory_mib

    def _process_memory(self):
        reserved = unstarted = 0.0
        counts = self.thread_budget.process_counts()
        for index, pool in enumerate(self.pools):
            count, missing = pool.process_reservation(counts[index])
            reserved += count * self._process_cost(index)
            unstarted += missing * self._process_cost(index)
        return reserved, unstarted

    def can_admit(self):
        """Check the proposed thread-budget reservation before launching engines."""
        if not self.adaptive_threads:
            return True

        def fits():
            retained, unstarted = self._process_memory()
            extra = self.thread_memory_mib * (self.thread_budget.total + sum(
                pool.search_thread_status()["idle_extra_search_threads"] for pool in self.pools))
            available = available_memory_mib()
            return (self.fixed_memory_mib + retained + extra <= self.budget_mib
                    and available is not None
                    and unstarted + self.thread_budget.total * self.thread_memory_mib
                    <= available - self.minimum_available_mib - self.MEMORY_MARGIN_MIB)

        if fits():
            return True
        # Old opponents' idle engines must not prevent useful new games.
        # Engines leased to running games retain their TT and thread count.
        for pool in self.pools:
            pool.close_idle()
        if fits():
            return True
        self.growth_blocked = True
        return False

    def rebalance(self, pending, active_counts, duration_weights, allow_growth=True,
                  *, completed_counts=None, target_counts=None):
        self.active_counts = active_counts
        changes = []
        for index, pool in enumerate(self.pools):
            if (not self.retired[index] and not active_counts[index]
                    and pending.remaining_for_player(index) == 0):
                closed = pool.retire()
                self.capacities[index] = 0
                self.retired[index] = True
                changes.append({"player": self.specs[index].name, "action": "retired", "closed_processes": closed})

        available = available_memory_mib() if allow_growth or self.adaptive_threads else None
        live_headroom = max(0.0, available - self.minimum_available_mib - self.MEMORY_MARGIN_MIB) if available is not None else 0.0
        retired_any = any(self.retired)
        old_capacities = list(self.capacities)
        growth_blocked = self.growth_blocked
        self.growth_blocked = False
        if self.adaptive_threads and allow_growth:
            for index, capacity in enumerate(self.capacities):
                if self.retired[index]:
                    continue
                target = min(self.max_match_sets, pending.remaining_for_player(index) + active_counts[index])
                if target > capacity:
                    self.capacities[index] = target
                    self.pools[index].grow(target)
        while not self.adaptive_threads and allow_growth and live_headroom > 0:
            candidates = []
            for index, capacity in enumerate(self.capacities):
                remaining = pending.remaining_for_player(index)
                if self.retired[index] or remaining == 0:
                    continue
                if capacity >= min(self.max_match_sets, remaining + active_counts[index]):
                    continue
                # Extra engines are useless without a free opponent process.
                if capacity > sum(self.capacities) - capacity:
                    continue
                extra_cost = 2 * self._process_cost(index)
                if extra_cost > min(self.budget_mib - self.reserved_memory_mib, live_headroom):
                    growth_blocked = True
                    continue
                work = (remaining + active_counts[index]) * max(0.001, duration_weights[index]) / capacity
                heavy = self.priorities[index] == 0 and self.specs[index].alpha != 1.0
                if retired_any:
                    rank = (int(heavy), work, -capacity, -self.costs[index], -index)
                else:
                    rank = (self.priorities[index], -capacity, work, -self.costs[index], -index)
                candidates.append((rank, index))
            if not candidates:
                break
            index = max(candidates)[1]
            self.capacities[index] += 1
            self.pools[index].grow(self.capacities[index])
            live_headroom -= 2 * self._process_cost(index)
        for index, old in enumerate(old_capacities):
            if self.capacities[index] > old:
                changes.append({"player": self.specs[index].name, "action": "increased",
                                "old_process_limit": old * 2, "new_process_limit": self.capacities[index] * 2})
        if changes:
            print("engine_allocation_changes", changes, flush=True)
        if changes or time.monotonic() - self.last_write >= 5:
            snapshot = {
                "sampled_at_unix_sec": time.time(),
                "initial_estimated_memory_mib": self.initial_memory_mib,
                "extra_memory_ceiling_mib": self.extra_memory_mib,
                "estimated_memory_budget_mib": self.budget_mib,
                "reserved_estimated_memory_mib": self.reserved_memory_mib,
                "minimum_available_memory_mib": self.minimum_available_mib,
                "adaptive_search_threads": self.adaptive_threads,
                "thread_budget": self.thread_budget.total if self.thread_budget else None,
                "allocated_search_threads": self.thread_budget.allocated_search_threads if self.thread_budget else None,
                "allocated_policy_processes": self.thread_budget.allocated_policy_processes if self.thread_budget else None,
                "allocated_player_processes": self.thread_budget.allocated_player_processes if self.thread_budget else None,
                "allocated_compute_slots": self.thread_budget.allocated if self.thread_budget else None,
                "budget_scope": "all_players_minimum_one_slot" if self.thread_budget else None,
                "allocation_policy": "fixed_total_between_games" if self.thread_budget else "fixed_per_process",
                "live_thread_changes": self.thread_budget.live_thread_changes if self.thread_budget else 0,
                "edax_thread_change_boundary": "between_moves_preserving_tt" if self.thread_budget else None,
                "process_allocation": "admitted_games_first" if self.adaptive_threads else "reserved_player_capacity",
                "admission_priority": "least_projected_completion_first" if self.adaptive_threads else "opening_order",
                "process_growth_limited_by_memory": growth_blocked,
                "players": [{"name": spec.name, "process_limit": self.capacities[i] * 2,
                             "created_processes": self.pools[i].created_processes,
                             "pending_sets": pending.remaining_for_player(i), "active_sets": active_counts[i],
                             "completed_sets": completed_counts[i] if completed_counts is not None else None,
                             "target_sets": target_counts[i] if target_counts is not None else None,
                             "retired": self.retired[i], "finish_priority": self.priorities[i],
                             "allocated_game_threads": self.thread_budget.player_threads(i) if self.thread_budget else [],
                             **(self.pools[i].search_thread_status() if self.adaptive_threads else {}),
                             "thread_change_pending": (self.pools[i].pending_thread_change
                                                       if self.adaptive_threads else False)}
                            for i, spec in enumerate(self.specs)],
            }
            if self.output_path is not None:
                atomic_write_json(self.output_path, snapshot)
            self.last_write = time.monotonic()
        return changes
