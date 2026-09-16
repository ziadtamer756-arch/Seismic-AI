from ai_prediction import predict_latest_earthquake



def test_prediction_structure():

    result = predict_latest_earthquake()


    assert "risk" in result

    assert "magnitude" in result

    assert "location" in result

    assert "depth" in result



def test_prediction_values():

    result = predict_latest_earthquake()


    assert result["risk"] in [

        "Low",

        "Medium",

        "High"

    ]


    assert isinstance(

        result["magnitude"],

        float

    )