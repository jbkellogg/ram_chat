import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.tools import StructuredTool, Tool
import re
import traceback
import config

# --- Core Scraping Logic (Using Requests - Modified Input) ---
def scrape_cate_athletics(sport_name: str) -> str:
    """
    Scrapes the Cate School athletics page for a specific sport using requests
    and BeautifulSoup. Finds schedule/results info from the HTML table.
    Accepts the sport name as a string.
    """
    # --- Simplified Input Handling ---
    if not sport_name or not isinstance(sport_name, str):
         return f"Error: Invalid input. Expected a sport name string, but received: {sport_name}"
    # --- END Input Handling ---

    base_url = "https://www.cate.org/athletics/"
    sport_slug = sport_name.lower().replace(" ", "-").replace("'", "")
    target_url = urljoin(base_url, sport_slug)

    print(f"Attempting to fetch with requests: {target_url}") # Verify URL

    # --- Fetch using Requests ---
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }
    html_content = None
    try:
        response = requests.get(target_url, headers=headers, timeout=20)
        response.raise_for_status()
        html_content = response.text
        print(f"Successfully retrieved page source via requests (length: {len(html_content)})")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred fetching {target_url}: {http_err}")
        status_code = response.status_code if 'response' in locals() and hasattr(response, 'status_code') else 'N/A'
        return f"Error: Could not fetch the athletics page. Status code: {status_code}. URL: {target_url}"
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred fetching {target_url}: {conn_err}")
        return f"Error: Could not connect to the server at {target_url}."
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout occurred fetching {target_url}: {timeout_err}")
        return f"Error: The request timed out while trying to reach {target_url}."
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request for {target_url}: {req_err}")
        return f"Error: An unexpected error occurred while fetching the page: {req_err}"
    except Exception as e:
        traceback.print_exc()
        return f"Error: An unexpected error occurred during requests fetching of {target_url}: {e}"

    if not html_content:
        return f"Error: Failed to retrieve HTML content from {target_url}."

    # --- Parse using BeautifulSoup ---
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        schedule_container = soup.find('div', id='section-4')
        if not schedule_container:
            print("Warning: Could not find schedule container 'div#section-4'.")
            content_area = soup.find('div', class_='fsPageContent') or soup.find('main') or soup
            text_content = re.sub(r'\s+', ' ', content_area.get_text(separator=' ', strip=True))
            if text_content:
                 return f"Could not find the specific schedule table container (div#section-4). General Page Content:\n{text_content[:1000]}..."
            else:
                 return f"Error: Could not find the schedule container (div#section-4) and no significant text content found on {target_url}."

        schedule_items = []
        print("Parsing schedule data from HTML table(s)...")
        team_columns = schedule_container.select('div.col-lg-6')
        if not team_columns:
            team_columns = [schedule_container]
            print("Warning: Could not find specific team columns (div.col-lg-6). Searching main container.")

        for col in team_columns:
            team_name_elem = col.find('strong')
            team_name = team_name_elem.get_text(strip=True) if team_name_elem else "Schedule"
            tables = col.select('table.schedule')
            print(f"Found {len(tables)} schedule table(s) for '{team_name}'.")

            for table in tables:
                tbody = table.find('tbody')
                if not tbody: continue
                rows = tbody.find_all('tr')
                print(f"  Parsing {len(rows)} rows for '{team_name}'.")
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) == 6:
                        day = cells[0].get_text(strip=True)
                        date_str = cells[1].get_text(strip=True)
                        opponent_cell = cells[2]
                        cancelled_tag = opponent_cell.find('strong', string=lambda t: t and 'Cancelled:' in t.upper())
                        is_cancelled = bool(cancelled_tag)
                        opponent = opponent_cell.get_text(separator=' ', strip=True).replace('Cancelled:', '').strip()
                        location_cell = cells[3]
                        location_link = location_cell.find('a')
                        location = location_link.get_text(strip=True) if location_link else location_cell.get_text(strip=True)
                        time_str = cells[4].get_text(strip=True)
                        result = cells[5].get_text(strip=True)
                        status = "(Cancelled) " if is_cancelled else ""
                        schedule_line = f"{team_name}: {status}{day} {date_str} vs {opponent} ({location} at {time_str})"
                        if result: schedule_line += f" - Result: {result}"
                        schedule_items.append(schedule_line)
                    else:
                        print(f"  Warning: Row in '{team_name}' table doesn't have 6 cells.")

        if not schedule_items:
            print("Warning: Found schedule container but could not extract any schedule items from tables.")
            return f"Found the schedule container on {target_url}, but couldn't parse any schedule items from the tables within it. The table structure might have changed."

        output = f"Athletics info for {sport_name.title()} from {target_url}:\n"
        output += "\nSchedule & Results:\n" + "\n".join([f"- {item}" for item in schedule_items]) + "\n"
        return output.strip()

    except Exception as parse_err:
        traceback.print_exc()
        return f"Error: An unexpected error occurred during HTML parsing of {target_url}: {parse_err}"


# --- Tool Creation Function (Modified) ---
def create_web_scraper_tool():
    """Creates the Langchain Tool for fetching and parsing Cate athletics pages."""
    web_tool = Tool(
        name="get_cate_athletics_info",
        func=scrape_cate_athletics,
        description=(
            "Use this tool ONLY for questions about CATE SCHOOL ATHLETICS schedules, scores, results, or team information. "
            "This tool scrapes the official Cate athletics website (cate.org/athletics/...) and is the **primary source for schedules and results**. "
            "**Input MUST be ONLY the specific sport name string** (e.g., 'girls lacrosse', 'boys water polo', 'football'). Do NOT wrap it in a dictionary. "
            "Use this alongside `email_retriever` to check for related announcements."
        ),
    )
    return web_tool

# --- Example usage (Modified) ---
if __name__ == '__main__':
    try:
        import brotli
        print("Brotli library is installed.")
    except ImportError:
        print("WARNING: Brotli library not found. Install using 'pip install brotli' if requests fails due to 'br' encoding.")

    test_sport = "girls lacrosse"
    print(f"\nTesting scraper for: {test_sport}")
    result = scrape_cate_athletics(test_sport)
    print("\nScraper Result:")
    print(result)
