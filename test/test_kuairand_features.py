import pandas as pd
import unittest
import numpy as np

from feature.kuairand_features import KuaiRandFeatureStore


class TestKuaiRandFeatures(unittest.TestCase):

    def test_extract_user_sequence_features(self):
        log = pd.read_csv("test/resources/kuairand-1k-test.csv")
        item_sequence, action_sequence = KuaiRandFeatureStore.extract_user_sequence_features(log)

        assert len(item_sequence.iloc[0]) == 176
        assert len(item_sequence.iloc[1]) == 338

        assert len(action_sequence.iloc[0]) == 176
        assert len(action_sequence.iloc[1]) == 338

        assert item_sequence.iloc[0][0] == 2528540
        assert action_sequence.iloc[0][0] == [1]
        assert item_sequence.iloc[0][1] == 4067506
        assert action_sequence.iloc[0][1] == [1, 7]

        # check the item that is clicked, liked and log view
        for i in range(len(item_sequence.iloc[1])):
            if item_sequence.iloc[1][i] == 4001361:
                assert action_sequence.iloc[1][i] == [1, 2, 7]

        print("KuaiRandFeatures test passed...")

    def test_extract_user_features(self):
        user_features = pd.read_csv("test/resources/user-features-test.csv")

        features = KuaiRandFeatureStore.extract_user_static_features(user_features)

        assert features.shape == (5, 3)
        assert features.iloc[0]["user_id"] == 0
        assert all(features.iloc[0]["numeric_features"] == [514, 150, 34, 799])
        assert all(features.iloc[1]["numeric_features"] == [457, 20, 3, 1474])

        assert all(
            features.iloc[0]["categorical_features"]
            == [1, 4, 7, 9, 17, 21, 28, 38, 40, 42, 77, 1047, 1570, 1584, 1618, 1635, 1874, 2199, 2203, 2205, 2210, 2213, 2214, 2216, 2218, 2220]
        )

    def test_extract_video_features(self):
        # Create test data for video_features_basic
        video_features_basic = pd.DataFrame(
            {
                "video_id": [1, 2, 3, 4, 5],
                "author_id": [101, 102, 103, 106, 100001],
                "music_id": [1001, 1002, 1003, 1006, 100001],
                "video_type": ["NORMAL", "AD", "UNKNOWN", "NORMAL", "AD"],
                "upload_dt": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"],
                "upload_type": ["LongImport", "ShortImport", "Web", "LongCamera", "UNKNOWN"],
                "tag": ["1,2,3", "4,5", None, "6,7,8,9", "10"],
                "visible_status": [1, 2, 3, np.nan, np.nan],
                "music_type": [1, 2, np.nan, 4, 5],
            }
        )

        # Create test data for video_features_statistic
        video_features_statistic = pd.DataFrame(
            {
                "video_id": [1, 2, 3, 4, 5],
                "view_count": [1000, 500, 2000, 800, 1500],
                "like_count": [100, 50, 200, 80, 150],
                "comment_count": [20, 10, 40, 15, 30],
                "share_count": [5, 2, 8, 3, 6],
            }
        )

        # Call the method
        result = KuaiRandFeatureStore.extract_video_features(video_features_basic, video_features_statistic)

        # Test basic structure
        assert result.shape == (5, 13)  # 5 rows, 12 columns (basic + statistic features)
        assert "video_id" in result.columns
        assert "video_type" in result.columns
        assert "upload_dt" in result.columns
        assert "upload_type" in result.columns
        assert "tag" in result.columns

        # Test categorical encoding
        # video_type: NORMAL=0, AD=1, UNKNOWN=2
        assert result.iloc[0]["video_type"] == 0  # NORMAL
        assert result.iloc[1]["video_type"] == 1  # AD
        assert result.iloc[2]["video_type"] == 2  # UNKNOWN

        # Test upload_type encoding (LongImport=0, ShortImport=2, Web=3, LongCamera=8, UNKNOWN=5)
        assert result.iloc[0]["upload_type"] == 0  # LongImport
        assert result.iloc[1]["upload_type"] == 2  # ShortImport
        assert result.iloc[2]["upload_type"] == 3  # Web
        assert result.iloc[3]["upload_type"] == 8  # LongCamera
        assert result.iloc[4]["upload_type"] == 5  # UNKNOWN

        # Test datetime conversion to unix timestamp
        expected_timestamp_1 = int(pd.to_datetime("2022-01-01").timestamp())
        assert result.iloc[0]["upload_dt"] == expected_timestamp_1

        # Test tag extraction
        assert result.iloc[0]["tag"] == [2, 3, 4]  # shifted by 1: 1->2, 2->3, 3->4
        assert result.iloc[1]["tag"] == [5, 6]  # shifted by 1: 4->5, 5->6
        assert result.iloc[2]["tag"] == [0]  # None becomes [0]
        assert result.iloc[3]["tag"] == [7, 8, 9, 10]  # shifted by 1
        assert result.iloc[4]["tag"] == [11]  # shifted by 1: 10->11

        # Test visible_status and music_type encoding
        assert result.iloc[0]["visible_status"] == 2
        assert result.iloc[1]["visible_status"] == 3
        assert result.iloc[2]["visible_status"] == 4
        assert result.iloc[3]["visible_status"] == 0
        assert result.iloc[4]["visible_status"] == 0

        assert result.iloc[0]["music_type"] == 2
        assert result.iloc[1]["music_type"] == 3
        assert result.iloc[2]["music_type"] == 0
        assert result.iloc[3]["music_type"] == 5
        assert result.iloc[4]["music_type"] == 6

        # Test that numeric features are normalized (after min-max, z-score, and log transformations)
        numeric_columns = ["view_count", "like_count", "comment_count", "share_count"]
        for col in numeric_columns:
            print(result[col])
            assert not result[col].isna().all()
            assert np.isfinite(result[col]).all()

        # Test that non-numeric features are preserved
        assert result.iloc[0]["video_id"] == 1
        assert result.iloc[1]["video_id"] == 2

        print("extract_video_features test passed...")
