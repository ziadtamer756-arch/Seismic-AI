from earthquake_data import get_recent_earthquakes
from database import create_database, save_earthquakes


print("Starting earthquake collector...")


create_database()


events = get_recent_earthquakes()


print(f"Found {len(events)} earthquakes")


save_earthquakes(events)


print("✅ Data saved to database")