/* Egaroucid Project, GPL-3.0-or-later. Search-independent Ponder selector.
 * Rules: reports/20261007_3_ggs_focused_ponder_protocol.md section 2.2.
 */
#pragma once
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

class Ponder_Focus {
public:
    struct Child {
        int policy;
        bool scheduled;
        int count = 0;
        bool complete = false;
        bool member = false;
        std::vector<double> values;
    };

    Ponder_Focus(std::vector<Child> children, int margin)
        : children_(std::move(children)), margin_(margin) {}

    int next() {
        int selected = -1;
        // Pass children get one probe; scheduled children get all six stages.
        for (int i = 0; i < static_cast<int>(children_.size()); ++i) {
            const Child &c = children_[i];
            if (!c.complete && c.count < (c.scheduled ? 6 : 1) &&
                (selected < 0 || c.count < children_[selected].count ||
                 (c.count == children_[selected].count && c.policy < children_[selected].policy))) {
                selected = i;
            }
        }
        if (selected >= 0) return selected;
        refresh();
        for (int i = 0; i < static_cast<int>(children_.size()); ++i) {
            const Child &c = children_[i];
            if (c.member && !c.complete && better(i, selected, true)) selected = i;
        }
        if (selected >= 0) return selected;
        // All members reached the allowed endgame limit. Use remaining time.
        for (int i = 0; i < static_cast<int>(children_.size()); ++i) {
            if (!children_[i].complete && better(i, selected, false)) selected = i;
        }
        return selected;
    }

    void record(int index, double value, bool complete = false) {
        Child &c = children_.at(index);
        c.values.push_back(value);
        ++c.count;
        c.complete = complete;
        // Update immediately, so the final membership is current even if
        // cancellation arrives before the next call to next().
        bool probed = std::all_of(children_.begin(), children_.end(), [](const Child &e) {
            return e.count >= (e.scheduled ? 6 : 1);
        });
        if (probed) refresh();
    }

    const Child &child(int index) const { return children_.at(index); }
    int frontier() const { return frontier_; }

private:
    std::vector<Child> children_;
    int margin_;
    int frontier_ = 0;

    double latest(int i) const {
        return children_[i].values.empty() ? -std::numeric_limits<double>::infinity() : children_[i].values.back();
    }
    bool better(int i, int selected, bool balance) const {
        if (selected < 0) return true;
        if (balance && children_[i].count != children_[selected].count)
            return children_[i].count < children_[selected].count;
        if (latest(i) != latest(selected)) return latest(i) > latest(selected);
        return children_[i].policy < children_[selected].policy;
    }
    double best(int stage, bool members_only) const {
        double value = -std::numeric_limits<double>::infinity();
        for (const Child &c : children_) {
            if (c.scheduled && (!members_only || c.member) && c.count >= stage)
                value = std::max(value, c.values[stage - 1]);
        }
        return value;
    }
    void refresh() {
        if (frontier_ == 0) {
            const double b5 = best(5, false), b6 = best(6, false);
            for (Child &c : children_) {
                c.member = c.scheduled && c.count >= 6 &&
                    !(b5 - c.values[4] > margin_ && b6 - c.values[5] > margin_);
            }
            frontier_ = 6;
        }
        while (true) {
            bool any = false, aligned = true;
            for (const Child &c : children_) {
                if (c.member) {
                    any = true;
                    aligned &= c.count >= frontier_ + 1;
                }
            }
            if (!any || !aligned) break;
            ++frontier_;
            // Compute both maxima before removing any member.
            const double previous = best(frontier_ - 1, true);
            const double current = best(frontier_, true);
            for (Child &c : children_) {
                if (c.member && previous - c.values[frontier_ - 2] > margin_ &&
                    current - c.values[frontier_ - 1] > margin_) c.member = false;
            }
            for (Child &c : children_) {
                if (c.scheduled && !c.member && !c.values.empty() &&
                    current - c.values.back() <= margin_) c.member = true;
            }
        }
    }
};
