import unittest

import pandas as pd

import wc2026_predictor as predictor


class HistoricalDataTests(unittest.TestCase):
    def test_fallback_dataset_contains_only_mens_world_cups(self):
        df = predictor.load_historical()

        womens_world_cup_years = {1991, 1995, 1999, 2003, 2007, 2011, 2015, 2019}
        self.assertTrue(womens_world_cup_years.isdisjoint(set(df["date"].dt.year)))
        usa_2019 = df[
            (df["date"].dt.year == 2019)
            & (
                (df["home_team"] == "United States")
                | (df["away_team"] == "United States")
            )
        ]
        self.assertTrue(usa_2019.empty)


class PoissonModelTests(unittest.TestCase):
    def test_small_samples_are_shrunk_toward_average(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-01"]),
                "home_team": ["Example"],
                "away_team": ["Opponent"],
                "home_score": [10],
                "away_score": [0],
                "tournament": ["FIFA World Cup"],
            }
        )
        model = predictor.PoissonModel(predictor.Elo())

        model.train(df, ["Example", "Missing"], years=12)

        self.assertEqual(model.sample_size["Example"], 1)
        self.assertLess(model.att["Example"], 2.2)
        self.assertGreater(model.att["Example"], predictor.AVG_GOALS)
        self.assertEqual(model.att["Missing"], predictor.AVG_GOALS)
        self.assertEqual(model.dfn["Missing"], predictor.AVG_GOALS)

    def test_training_window_is_anchored_to_latest_observation(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["1999-01-01", "2000-01-01"]),
                "home_team": ["Old Team", "Current Team"],
                "away_team": ["Opponent", "Opponent"],
                "home_score": [5, 2],
                "away_score": [0, 1],
                "tournament": ["Friendly", "Friendly"],
            }
        )
        model = predictor.PoissonModel(predictor.Elo())

        model.train(df, ["Current Team"], years=1)

        self.assertEqual(model.sample_size["Current Team"], 1)


if __name__ == "__main__":
    unittest.main()
