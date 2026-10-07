/* Egaroucid Project, GPL-3.0-or-later. No engine searches are needed. */
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "../../engine/ponder_focus.hpp"

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void probe(Ponder_Focus &focus, const std::vector<double> &values) {
    while (focus.frontier() == 0) {
        int i = focus.next();
        require(i >= 0, "probe must select a child");
        focus.record(i, values.at(i));
    }
}
void tests() {
    // Probe by count, then policy (vector order deliberately differs).
    Ponder_Focus a({{8,true},{3,true},{12,true}}, 2);
    for (int stage = 0; stage < 6; ++stage) {
        for (int i : {1,0,2}) {
            require(a.next() == i, "probe ordering");
            a.record(i, 0);
        }
    }
    require(a.frontier() == 6, "probe frontier");
    // A stable three-disc advantage isolates one child.
    Ponder_Focus b({{1,true},{2,true},{3,true}}, 2);
    probe(b, {5,2,1});
    for (int k = 0; k < 6; ++k) {
        require(b.next() == 0, "unique best child");
        b.record(0,5);
    }
    // At the inclusive margin boundary, both children progress together.
    Ponder_Focus c({{1,true},{2,true}}, 2);
    probe(c, {5,3});
    for (int k = 0; k < 24; ++k) {
        int i = c.next();
        c.record(i, i == 0 ? 5 : 3);
        require(std::abs(c.child(0).count - c.child(1).count) <= 1, "balance");
        require(c.child(0).member && c.child(1).member, "inclusive margin");
    }
    // One bad stage is retained; the second consecutive bad stage removes it.
    Ponder_Focus d({{1,true},{2,true}}, 2);
    probe(d, {5,4});
    for (int k = 0; k < 2; ++k) { int i = d.next(); d.record(i, i == 0 ? 5 : 1); }
    require(d.child(1).member, "one bad stage retained");
    for (int k = 0; k < 2; ++k) { int i = d.next(); d.record(i, i == 0 ? 5 : 1); }
    require(!d.child(1).member, "two bad stages removed");
    // Probe exclusion also requires both stage 5 and stage 6 to be bad.
    Ponder_Focus e({{1,true},{2,true}}, 2);
    for (int stage = 1; stage <= 6; ++stage) {
        for (int i : {0,1}) {
            require(e.next() == i, "initial probe");
            e.record(i, i == 0 ? 5 : stage == 6 ? 1 : 4);
        }
    }
    require(e.child(1).member, "one bad probe stage retained");
    // A drop of the active best reintroduces the excluded child and catches up.
    require(b.next() == 0, "best before drop");
    b.record(0,2);
    require(b.child(1).member && b.child(2).member, "reentry after best drops");
    require(b.next() == 1, "lagging child catches up first");
    for (int k = 0; k < 8; ++k) {
        int i = b.next();
        require(i != 0, "no leading child until others catch up");
        b.record(i,2);
    }
    // Completed active children leave excluded children available to search.
    Ponder_Focus f({{1,true},{2,true}}, 2);
    probe(f, {5,1});
    require(f.next() == 0, "active first");
    f.record(0,5,true);
    require(f.next() == 1, "excluded after active completion");
    f.record(1,1,true);
    require(f.next() == -1, "all complete");
    // Pass child receives exactly one probe and no active visits.
    Ponder_Focus g({{1,true},{2,false}}, 2);
    probe(g, {5,10});
    require(g.child(1).count == 1, "pass probed once");
    require(!g.child(1).member, "pass never becomes a member");
    for (int k = 0; k < 4; ++k) {
        require(g.next() == 0, "pass waits for active completion");
        g.record(0,5,k == 3);
    }
    require(g.next() == 1, "pass available after completion");
    // Ties after probing use raw latest value, then policy.
    Ponder_Focus h({{8,true},{3,true}}, 2);
    probe(h,{4,4});
    require(h.next() == 1, "policy tie break");
    h.record(1,4);
    require(h.next() == 0, "count takes precedence");
}
}
int main() {
    try { tests(); }
    catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
    std::cout << "PASS: probe, concentration, balance, hysteresis, reentry, completion, pass, ties\n";
}
