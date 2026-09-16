import pandas as pd

from ai_prediction import prepare_features



def test_feature_generation():


    earthquake = pd.Series(

        {

        "depth":10,

        "latitude":30,

        "longitude":40,

        "time":1700000000000,

        "magnitude":4.5,

        "place":"test"

        }

    )



    result = prepare_features(

        earthquake

    )


    assert result.shape[1] == 12