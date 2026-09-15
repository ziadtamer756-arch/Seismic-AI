import folium


def create_map(df):

    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="OpenStreetMap"
    )


    if len(df) > 0:


        for _, row in df.iterrows():


            try:

                magnitude = float(row["magnitude"])

                radius = magnitude * 3


                folium.CircleMarker(

                    location=[
                        row["latitude"],
                        row["longitude"]
                    ],


                    radius=radius,


                    popup=f"""
                    Location: {row.get('place','Unknown')}<br>
                    Magnitude: {magnitude}<br>
                    Depth: {row.get('depth','Unknown')} km
                    """,


                    color="red",
                    fill=True

                ).add_to(m)


            except Exception:

                continue



    return m