from bs4 import BeautifulSoup
import requests
import smtplib


url = "https://www.amazon.com/dp/B075CYMYK6?ref_=cm_sw_r_cp_ud_ct_FM9M699VKHTT47YD50Q6&th=1"

header = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Priority": "u=0, i",
    "Sec-Ch-Ua": "\"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"Windows\"",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
  }

responce = requests.get(url,headers=header)
responce.raise_for_status()
text = responce.text

EMAIL_ADDRESS="shryesPython19"
EMAIL_PASSWORD="vrrsuydjhnzxleub"


soup = BeautifulSoup(text,"html.parser")
print(soup.prettify())
v = soup.find(name="span",class_="a-price-whole").getText()
v = float( v + soup.find(name="span",class_="a-price-fraction").getText())

message = soup.find(name="span",id="productTitle").getText()

if v <= 99.0:
    with smtplib.SMTP("smtp.gmail.com") as connection:
        connection.starttls()
        result = connection.login(user=EMAIL_ADDRESS,password=EMAIL_PASSWORD)
        connection.sendmail(
            from_addr=EMAIL_ADDRESS,
            to_addrs="shryes107@gmail.com",
            msg=f"Subject:Amazon Price Alert!\n\n{message}\n{url}".encode("utf-8")
        )