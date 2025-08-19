# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from bs4 import BeautifulSoup
#
# def scrape_page(url):
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     driver = webdriver.Chrome(options=options)
#
#     driver.get(url)
#     html = driver.page_source
#     soup = BeautifulSoup(html, 'html.parser')
#     driver.quit()
#
#     main_content = soup.find('div', class_='main-content') or soup.find('main')
#     if not main_content:
#         print("Main content not found")
#         return ""
#
#     return main_content.get_text(separator="\n").strip()


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_page(url):
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        driver.get(url)
        driver.implicitly_wait(10)  # Wait for JS to render
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
        return f"[ERROR] Selenium scraping failed: {str(e)}"
