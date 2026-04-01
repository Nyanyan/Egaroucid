# from https://blog.amedama.jp/entry/elo-rating
# modified by Takuto Yamana

# レーティングが更新される大きさを表す K ファクター
ELO_K_FACTOR = 16


class Elo_player:
    """プレイヤーを表現したクラス"""

    def __init__(self, rating=1500):
        self.rating = rating

    def win_proba(self, other_player: "Elo_player") -> float:
        """他のプレイヤーに勝利する確率を計算するメソッド"""
        return 1. / (10. ** ((other_player.rating - self.rating) / 400.) + 1.)

    def get_rating(self):
        return self.rating

def update_rating(winner: Elo_player, loser: Elo_player) -> tuple[Elo_player, Elo_player]:
    """対戦成績を元にレーティングを更新する"""
    new_winner_rating = winner.rating + ELO_K_FACTOR * loser.win_proba(winner)
    new_loser_rating = loser.rating - ELO_K_FACTOR * loser.win_proba(winner)
    return Elo_player(new_winner_rating), Elo_player(new_loser_rating)


def update_rating_draw(player1: Elo_player, player2: Elo_player) -> tuple[Elo_player, Elo_player]:
    """引き分け時の対戦成績を元にレーティングを更新する"""
    new_player1_rating = player1.rating + ELO_K_FACTOR * (0.5 - player1.win_proba(player2))
    new_player2_rating = player2.rating + ELO_K_FACTOR * (0.5 - player2.win_proba(player1))
    return Elo_player(new_player1_rating), Elo_player(new_player2_rating)

