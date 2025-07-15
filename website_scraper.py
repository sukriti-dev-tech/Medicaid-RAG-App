#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.request import Request, urlopen
import ssl
from io import StringIO
from urllib import request
from io import StringIO, BytesIO
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import pdf_chunker
from collections import defaultdict


# In[2]:


class webScraper:
    def __init__(self, name):
            self.name = name

    def getWebsitePdfUrls(self,chunker) -> list[str]:
        driver = webdriver.Chrome()
        # Wait for an element to be present
        assert "No results found." not in driver.page_source
        #driver.implicitly_wait(10)

        driver.get('https://ldh.la.gov/page/1681')
        try:
            elements = WebDriverWait(driver, 100).until(
            EC.presence_of_element_located((By.TAG_NAME, "ul"))
            )
            #for listobj in elements:
            print("Wait finished, reading URLs now")
            pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")

            print("Read no. of links=",len(pdf_links))
            pdf_urls = []
            for pdf_link in pdf_links:
                pdf_urls.append(pdf_link.get_attribute("href"))
                print("Reading from URL = ",pdf_link.get_attribute("href"))
            #pdf_urls = ["https://ldh.la.gov/assets/medicaid/MedicaidEligibilityPolicy/I-1630.pdf"]
            all_final_documents = chunker.process_pdfs(pdf_urls)

            if all_final_documents:
                print("\n\n--- Example of Final Documents ---")
                docs_by_file = defaultdict(list)
                for doc in all_final_documents:
                    docs_by_file[doc.metadata['file_name']].append(doc)

                for file_name, docs in docs_by_file.items():
                    print(f"\n--- Results for: '{file_name}' ({len(docs)} documents) ---")
                    # Show the metadata header from the first document's content
                    header = "\n".join(docs[0].page_content.split('\n')[:3])
                    print("Header from first document's content:")
                    print(header)
                    print("-" * 20)

            return all_final_documents
        finally:
            driver.quit()
            print("Scraping from URLs is complete")


# In[3]:


#consolidated_chunks = webScraper().getWebsitePdfUrls("just")
#print ("Finished scraping, no. of chunks fetched = ",len(consolidated_chunks))

