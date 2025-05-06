import requests
import json
from datetime import date, datetime

def get_flik_menu_api(school_subdomain: str, school_identifier: str, meal_type: str, target_date: date) -> dict | None:
    """
    Fetches the menu for a specific school, meal type, and date using the Flik Dining API.

    Args:
        school_subdomain: The subdomain for the school (e.g., 'cate').
        school_identifier: The identifier string for the school (e.g., 'cate-school-high-school').
        meal_type: The meal type (e.g., 'dinner', 'lunch', 'breakfast'). Case-sensitive for the API.
        target_date: A datetime.date object for the desired menu date.

    Returns:
        A dictionary containing the menu data for the specified date from the API response,
        or None if the request fails or data for the specific date isn't found in the response.
        Note: The API endpoint includes '/weeks/', so the response might contain data
        for the entire week. This function attempts to extract only the target date's data.
    """
    # --- Use target_date for URL construction ---
    year = target_date.year
    month = f"{target_date.month:02d}" # Ensure two digits (e.g., 05)
    day = f"{target_date.day:02d}"   # Ensure two digits (e.g., 06)
    # --- End change ---

    # Construct the API URL using the target date components
    api_url = f"https://{school_subdomain}.api.flikisdining.com/menu/api/weeks/school/{school_identifier}/menu-type/{meal_type}/{year}/{month}/{day}/"

    print(f"Requesting Flik menu from API: {api_url}") # This will now show the correct target date in the URL

    try:
        response = requests.get(api_url, timeout=20) # Add a reasonable timeout
        response.raise_for_status() # Raise an HTTPError for bad status codes (4xx or 5xx)

        print("API request successful.")
        full_week_data = response.json()

        # --- Extract the menu for the specific target date ---
        target_date_str = target_date.strftime('%Y-%m-%d') # Standard date format often used in APIs
        menu_for_day = None

        if 'days' in full_week_data and isinstance(full_week_data['days'], list):
            for day_data in full_week_data['days']:
                if day_data.get('date') == target_date_str:
                    menu_for_day = day_data
                    break
        else:
            print("Warning: Unexpected API response structure. Could not find 'days' list.")
            # Save the full response for debugging
            try:
                debug_filename = f"debug_api_response_unexpected_{target_date_str}.json"
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(full_week_data, f, indent=2)
                print(f"Saved full unexpected API response to {debug_filename}")
            except Exception as save_err:
                 print(f"Could not save full API response: {save_err}")
            return None


        if menu_for_day:
            print(f"Successfully extracted menu data for {target_date_str}.")
            return menu_for_day
        else:
            print(f"Error: Could not find menu data for the specific date {target_date_str} within the API response.")
            # Save the full response for debugging if the specific day wasn't found
            try:
                debug_filename = f"debug_api_response_date_not_found_{target_date_str}.json"
                with open(debug_filename, "w", encoding="utf-8") as f:
                    json.dump(full_week_data, f, indent=2)
                print(f"Saved full API response to {debug_filename}")
            except Exception as save_err:
                 print(f"Could not save full API response: {save_err}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
        print(f"Response Text: {response.text[:500]}...")
        return None
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
        return None
    except requests.exceptions.Timeout as timeout_err:
        print(f"Request timed out: {timeout_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the API request: {req_err}")
        return None
    except json.JSONDecodeError:
        print("Failed to decode JSON response from API. The response might not be valid JSON.")
        # Save the raw response text for debugging
        try:
            debug_filename = f"debug_api_response_raw_{target_date.strftime('%Y-%m-%d')}.txt"
            with open(debug_filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Saved raw API response text to {debug_filename}")
        except Exception as save_err:
            print(f"Could not save raw API response: {save_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- Main execution example ---
if __name__ == "__main__":
    # --- Configuration ---
    school_subdomain = "cate"
    school_identifier = "cate-school-high-school"
    # Choose meal: 'breakfast', 'lunch', 'dinner', etc. (check API for exact names)
    meal = "dinner"
    # Set the desired date
    # target_dt = date.today() # Get today's menu
    #target_dt = date(2025, 5, 13) # Or specify a date
    target_dt = date.today() # Or specify a date

    print(f"\n--- Getting Flik Menu via API ---")
    print(f"School: {school_identifier} ({school_subdomain})")
    print(f"Meal:   {meal.capitalize()}")
    print(f"Date:   {target_dt.strftime('%A, %B %d, %Y')}") # Nicely formatted date
    print("-" * 30)

    # Call the function
    menu_data = get_flik_menu_api(school_subdomain, school_identifier, meal, target_dt)

    # --- Output Results ---
    if menu_data:
        print("\n--- Successfully Retrieved Menu Data ---")
        # ... (optional printing of full JSON) ...

        if 'menu_items' in menu_data and isinstance(menu_data['menu_items'], list): # Add check if menu_items is a list
            print("\n--- Menu Items ---")
            for item_info in menu_data['menu_items']:
                # Ensure item_info is a dictionary before proceeding
                if not isinstance(item_info, dict):
                    print(f"Warning: Skipping non-dictionary item in menu_items: {item_info}")
                    continue

                # Safely get the food dictionary, defaulting to an empty dict if 'food' is missing or None
                food_dict = item_info.get('food') or {} # Handles 'food': null case
                food_name = food_dict.get('name', 'Name N/A')

                # Safely get the station dictionary
                station_dict = item_info.get('menu_station') or {} # Handles 'menu_station': null case
                station_name = station_dict.get('station', 'Station N/A')

                print(f"- {station_name}: {food_name}")
        elif 'menu_items' not in menu_data:
             print("Warning: 'menu_items' key not found in the retrieved menu data.")
        else:
             print(f"Warning: 'menu_items' key exists but is not a list: {type(menu_data['menu_items'])}")


    else:
        print("\n--- Failed to retrieve or process menu data for the specified date. ---")
        print("Check debug files (if created) or error messages above.")

