import requests
from bs4 import BeautifulSoup

def get_lowest_price(card_name, pack_name, condition, edition, country='Germany'):
    # Constructing the search URL
    search_url = f'https://www.cardmarket.com/en/YuGiOh/Cards/{card_name.replace(" ", "-")}/{pack_name}?country={country}&condition={condition}&edition={edition}'

    # Sending a GET request to the URL
    response = requests.get(search_url)

    # Parsing the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Finding the lowest offer price
    lowest_price_tag = soup.find('td', class_='article-3')
    if lowest_price_tag:
        lowest_price = lowest_price_tag.text.strip()
        return lowest_price
    else:
        return "No offers found."

if __name__ == "__main__":
    card_name = input("Enter the name of the Yu-Gi-Oh card: ")
    pack_name = input("Enter the pack name: ")
    condition = input("Enter the condition (NM, EX, GD, LP, PL, PO): ")
    edition = input("Enter the edition (1st Edition, Unlimited Edition): ")
    lowest_price = get_lowest_price(card_name, pack_name, condition, edition)
    print(f"The lowest offer price for {card_name} ({edition}, {condition}) from vendors in Germany on cardmarket.eu is: {lowest_price}")

