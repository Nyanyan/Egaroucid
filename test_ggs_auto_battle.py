import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parent / "bin" / "ggs" / "ggs_auto_battle.py"
SPEC = importlib.util.spec_from_file_location("ggs_auto_battle", MODULE_PATH)
assert SPEC is not None
ggs_auto_battle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ggs_auto_battle
SPEC.loader.exec_module(ggs_auto_battle)


class GgsAutoBattleStatisticsTest(unittest.TestCase):
    def test_parse_match_result_when_own_player_is_first(self):
        line = (
            "\x1b[32mGGS RECV> /os: - match .28 2613 egrcd "
            "2646 nyanyan s8r14 R -1.00  .82387"
        )

        result = ggs_auto_battle.parse_ggs_match_result(line, "egrcd", "nyanyan")

        self.assertIsNotNone(result)
        self.assertEqual(".28", result.match_id)
        self.assertEqual(-1.0, result.disc_difference)

    def test_parse_match_result_when_own_player_is_second(self):
        line = (
            "GGS RECV> /os: - match .99 2646 nyanyan "
            "2613 egrcd s8r14 R +3.00  .82387"
        )

        result = ggs_auto_battle.parse_ggs_match_result(line, "egrcd", "nyanyan")

        self.assertIsNotNone(result)
        self.assertEqual(".99", result.match_id)
        self.assertEqual(-3.0, result.disc_difference)

    def test_parse_match_result_ignores_other_opponents(self):
        line = "GGS RECV> /os: - match .42 2613 egrcd 2400 other s8r14 R +1.00 .1"

        result = ggs_auto_battle.parse_ggs_match_result(line, "egrcd", "nyanyan")

        self.assertIsNone(result)

    def test_detects_player_not_accepting_error(self):
        line = (
            "\x1b[33mGGS INFO> server error: /os: "
            "ERR Player is not accepting new matches."
        )

        self.assertTrue(ggs_auto_battle.is_player_not_accepting_error(line))

    def test_ignores_other_server_errors_for_retry(self):
        line = "GGS INFO> server error: /os: ERR Rated game request already exists."

        self.assertFalse(ggs_auto_battle.is_player_not_accepting_error(line))

    def test_statistics_treat_draw_as_half_win(self):
        statistics = ggs_auto_battle.MatchStatistics()

        statistics.record(ggs_auto_battle.MatchResult(".1", 2.0))
        statistics.record(ggs_auto_battle.MatchResult(".2", 0.0))
        statistics.record(ggs_auto_battle.MatchResult(".3", -4.0))

        self.assertEqual(1, statistics.wins)
        self.assertEqual(1, statistics.draws)
        self.assertEqual(1, statistics.losses)
        self.assertEqual(0.5, statistics.win_rate())
        self.assertAlmostEqual(-2.0 / 3.0, statistics.average_disc_difference())

    def test_format_signed_score_normalizes_negative_zero(self):
        self.assertEqual("+0.00", ggs_auto_battle.format_signed_score(-0.0))


if __name__ == "__main__":
    unittest.main()
