import traceback
import json  # Import json for parsing
from datetime import date, timedelta, datetime
from langchain.tools import Tool  # Use base Tool class
import config

try:
    from get_flik_menus import get_flik_menu_api
except ImportError as import_err:
    print(f"ERROR: Could not import get_flik_menu_api: {import_err}")
    # Define a dummy function if import fails
    def get_flik_menu_api(*args, **kwargs):
        return {"error": "Flik API function not available due to import error."}

# --- Tool Wrapper Function (Modified for single string input) ---
def get_menu_tool_wrapper(input_str: str) -> str:
    """
    Gets the Flik menu for a given meal and date, handling date calculation and formatting.
    Accepts a JSON string containing 'meal_type' and 'date_str'.
    """
    print(f"\n--- Executing Menu Tool Wrapper ---")
    print(f"  Raw input_str received: '{input_str}' (Type: {type(input_str)})")

    # --- Parse JSON Input ---
    try:
        input_data = json.loads(input_str)
        meal_type = input_data.get("meal_type")
        # Handle potential missing date_str, default to 'today'
        date_str = input_data.get("date_str", "today")
        print(f"  Parsed meal_type: '{meal_type}'")
        print(f"  Parsed date_str: '{date_str}'")
    except json.JSONDecodeError:
        print(f"  Error: Invalid JSON input string: {input_str}")
        return "Error: Tool input must be a valid JSON string containing 'meal_type' and optionally 'date_str'."
    except Exception as parse_err:
        print(f"  Error parsing input JSON: {parse_err}")
        return f"Error processing input: {parse_err}"
    # --- End JSON Parsing ---

    if not meal_type or not isinstance(meal_type, str):
        print("  Error: Invalid or missing 'meal_type' in JSON.")
        return "Error: Missing or invalid 'meal_type' in JSON input."
    if not date_str or not isinstance(date_str, str):
        print("  Error: Invalid or missing 'date_str' in JSON.")
        return "Error: Missing or invalid 'date_str' in JSON input. Use 'today', 'tomorrow', or 'YYYY-MM-DD'."

    meal_type_lower = meal_type.lower()
    date_str_lower = date_str.lower()
    target_dt = None

    try:
        # --- Date Calculation ---
        print("  Attempting date calculation...")
        if date_str_lower == "today":
            target_dt = date.today()
            print(f"    Calculated date as 'today': {target_dt}")
        elif date_str_lower == "tomorrow":
            target_dt = date.today() + timedelta(days=1)
            print(f"    Calculated date as 'tomorrow': {target_dt}")
        else:
            try:
                # Attempt to parse YYYY-MM-DD format
                target_dt = datetime.strptime(date_str, '%Y-%m-%d').date()
                print(f"    Parsed date string '{date_str}' as: {target_dt}")
            except ValueError as date_err:
                print(f"    Error parsing date string '{date_str}': {date_err}")
                return f"Error: Invalid date format '{date_str}'. Please use 'today', 'tomorrow', or 'YYYY-MM-DD'."
        # --- End Date Calculation ---

        # --- Check if target_dt was successfully assigned ---
        if target_dt is None:
            print("  Error: target_dt was not assigned after date calculation logic.")
            return "Internal Error: Could not determine the target date."
        # --- End Check ---

        print(f"  Calling Flik API function (get_flik_menu_api) for {meal_type_lower} on {target_dt.isoformat()}...")
        # --- Call the actual API function ---
        menu_data = get_flik_menu_api(
            school_subdomain=config.FLIK_SCHOOL_SUBDOMAIN,
            school_identifier=config.FLIK_SCHOOL_IDENTIFIER,
            meal_type=meal_type_lower,
            target_date=target_dt  # Pass the date object
        )
        print(f"  Flik API function returned.")  # Log after call returns
        # --- End API Call ---

        # --- Process API Response ---
        print("  Processing API response...")
        if not menu_data:
            print("    API returned None or empty data.")
            return f"Could not retrieve menu data from Flik API for {meal_type_lower} on {target_dt.isoformat()}."

        # Handle potential error message from dummy function or API function itself
        if isinstance(menu_data, dict) and menu_data.get("error"):
            print(f"    API function returned an error: {menu_data['error']}")
            return f"Error calling Flik API function: {menu_data['error']}"

        # Check the structure before accessing 'menu_items'
        if isinstance(menu_data, dict) and 'menu_items' in menu_data and isinstance(menu_data['menu_items'], list):
            print(f"    Found {len(menu_data['menu_items'])} items in 'menu_items'.")
            output_lines = [f"Menu for {meal_type.capitalize()} on {target_dt.strftime('%A, %B %d, %Y')}:"]
            items_by_station = {}
            for item_info in menu_data['menu_items']:
                if not isinstance(item_info, dict):
                    continue
                food_dict = item_info.get('food') or {}
                food_name = food_dict.get('name', 'N/A')
                station_dict = item_info.get('menu_station') or {}
                station_name = station_dict.get('station', 'Uncategorized')

                if station_name not in items_by_station:
                    items_by_station[station_name] = []
                if food_name != 'N/A':
                    items_by_station[station_name].append(food_name)

            if not items_by_station:
                print("    No valid items extracted after processing 'menu_items'.")
                return f"No menu items found listed for {meal_type_lower} on {target_dt.isoformat()} in the retrieved data."

            for station, items in items_by_station.items():
                output_lines.append(f"\n  {station}:")
                for item in items:
                    output_lines.append(f"    - {item}")
            final_output = "\n".join(output_lines)
            print("    Successfully formatted menu output.")
            return final_output
        else:
            # Log the structure if 'menu_items' is missing or wrong type
            response_type = type(menu_data).__name__
            keys = menu_data.keys() if isinstance(menu_data, dict) else "N/A (Not a dict)"
            print(f"    Warning: 'menu_items' key missing or not a list in API response. Response type: {response_type}, Keys: {keys}")
            return f"Menu data retrieved for {target_dt.isoformat()}, but no valid 'menu_items' list was found."
        # --- End Process API Response ---

    except Exception as e:
        # --- Error Logging ---
        print(f"--- ERROR in menu tool wrapper ---")
        print(f"  Exception Type: {type(e).__name__}")
        print(f"  Exception Args: {e.args}")
        traceback.print_exc()  # Print full traceback to console
        return f"An unexpected error occurred while trying to get the menu: {e}"

# --- Tool Creation Function (Modified) ---
def create_menu_tool():
    """Creates the Langchain Tool for the Flik dining menu."""
    print("--- Creating Menu Tool (Basic Tool Class) ---")
    menu_tool = Tool(
        name="get_dining_menu",
        func=get_menu_tool_wrapper,  # Pass the wrapper function
        description=(
            "Use this tool ONLY for questions about the food menu for a specific meal (breakfast, lunch, or dinner) on a specific date. "
            "**Input MUST be a JSON string** containing 'meal_type' (e.g., 'lunch') and 'date_str'. "
            "For 'date_str', use 'today', 'tomorrow', or a specific date in 'YYYY-MM-DD' format. "
            "Example Input: '{{\"meal_type\": \"lunch\", \"date_str\": \"2025-05-06\"}}' or '{{\"meal_type\": \"dinner\", \"date_str\": \"today\"}}'. "
            "Determine the meal type and the specific date. If the user says 'today' or 'tomorrow', use those exact words for 'date_str'. "
            "For all other dates (like 'Wednesday', 'next Friday', 'May 10th'), calculate the exact date in YYYY-MM-DD format and use that for 'date_str'. "
            "If the user asks a follow-up question about a menu (e.g., 'is it vegetarian?') and doesn't specify a date, check the chat history to see which date was discussed previously and use that date in the JSON input."
        ),
    )
    return menu_tool