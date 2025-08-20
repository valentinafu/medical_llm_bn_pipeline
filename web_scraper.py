from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# This script provides a reliable way to scrape the visible text of a web page,even for sites that require JavaScript rendering:
# - Uses Selenium with Chrome WebDriver in headless mode to load a given URL.
# - Extracts the full page html content.
# - Parses the html with BeautifulSoup to locate the main content area
# - Returns the text content from the main section of the page.
# - If scraping fails, it returns an error message.

def scrape_page(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        driver.get(url)
        driver.implicitly_wait(10)
        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')

        main_content = (
            soup.find('div', class_='main-content') or
            soup.find('main') or
            soup.find('div', {'id': 'content'}) or
            soup.body
        )

        if not main_content:
            print("Main content not found.")
            return ""

        return main_content.get_text(separator="\n", strip=True)

    except Exception as e:
        return f"[ERROR] Selenium scraping on current page failed: {str(e)}"
