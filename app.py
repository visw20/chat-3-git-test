
# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }


# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message
    
#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         general_query_mode = False
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Thank you. Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)
        
#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
    
#     return jsonify({"response": res})


# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)



#27/08/24
#Database

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }


# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn

# # # Function to save user interaction in the database
# # def save_interaction(interaction_type, user_input):
# #     try:
# #         conn = get_db_connection()
# #         cursor = conn.cursor()
# #         # Assuming the table has columns named 'general_queries', 'feedback', and 'order_related_issues'
# #         if interaction_type == 'general_queries':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (general_queries)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'feedback':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (feedback)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'order_related_issues':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (order_related_issues)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         else:
# #             # Handle unknown interaction types if necessary
# #             print(f"Unknown interaction type: {interaction_type}")
# #             return jsonify({"response": "An unknown error occurred. Please try again later."})
        
# #         conn.commit()
# #     except Exception as e:
# #         print(f"Error saving interaction: {e}")
# #         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
# #     finally:
# #         cursor.close()
# #         conn.close()

# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         # Adjusted to match the actual interaction types
#         if interaction_type == 'general_query':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (general_queries)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         elif interaction_type == 'feedback':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (feedback)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         elif interaction_type == 'order_related_issue':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (order_related_issues)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         else:
#             # Handle unknown interaction types if necessary
#             print(f"Unknown interaction type: {interaction_type}")
#             return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()
#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()


# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message
    
#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Thank you. Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)
        
#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
    
#     return jsonify({"response": res})


# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)










# #28/08/24
# #database

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }


# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn


# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         # Adjusted to match the actual interaction types
#         if interaction_type == 'general_query':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (general_queries)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         elif interaction_type == 'feedback':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (feedback)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         elif interaction_type == 'order_related_issue':
#             insert_query = sql.SQL("""
#                 INSERT INTO chatbot (order_related_issues)
#                 VALUES (%s)
#             """)
#             cursor.execute(insert_query, (user_input,))
#         else:
#             # Handle unknown interaction types if necessary
#             print(f"Unknown interaction type: {interaction_type}")
#             return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()
#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()


# #one more type for save the feedback, general queries , order related issues
# # Function to save user interaction in the database
# # def save_interaction(interaction_type, user_input):
# #     try:
# #         conn = get_db_connection()
# #         cursor = conn.cursor()

# #         # Determine which column to insert the data into based on interaction type
# #         if interaction_type == 'general_query':
# #             column_name = 'general_queries'
# #         elif interaction_type == 'feedback':
# #             column_name = 'feedback'
# #         elif interaction_type == 'order_related_issue':
# #             column_name = 'order_related_issues'
# #         else:
# #             print(f"Unknown interaction type: {interaction_type}")
# #             return jsonify({"response": "An unknown error occurred. Please try again later."})

# #         # Dynamically create the SQL insert statement based on the column name
# #         insert_query = sql.SQL("""
# #             INSERT INTO chatbot ({})
# #             VALUES (%s)
# #         """).format(sql.Identifier(column_name))

# #         cursor.execute(insert_query, (user_input,))
# #         conn.commit()

# #     except Exception as e:
# #         print(f"Error saving interaction: {e}")
# #         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
# #     finally:
# #         cursor.close()
# #         conn.close()


# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message
    
#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Thank you. Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)
        
#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
    
#     return jsonify({"response": res})


# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)








# #29/08/24
# #database

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }



# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn


# # # Function to save user interaction in the database
# # def save_interaction(interaction_type, user_input):
# #     try:
# #         conn = get_db_connection()
# #         cursor = conn.cursor()
# #         # Adjusted to match the actual interaction types
# #         if interaction_type == 'general_query':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (general_queries)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'feedback':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (feedback)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'order_related_issue':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (order_related_issues)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         else:
# #             # Handle unknown interaction types if necessary
# #             print(f"Unknown interaction type: {interaction_type}")
# #             return jsonify({"response": "An unknown error occurred. Please try again later."})
        
# #         conn.commit()
# #     except Exception as e:
# #         print(f"Error saving interaction: {e}")
# #         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
# #     finally:
# #         cursor.close()
# #         conn.close()

# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Check if there is an existing row that needs to be updated
#         select_query = """
#             SELECT interaction_id FROM chatbot 
#             WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
#             ORDER BY interaction_id DESC LIMIT 1
#         """
#         cursor.execute(select_query)
#         result = cursor.fetchone()
        
#         # Determine if we should insert a new row or update an existing one
#         if result:
#             interaction_id = result[0]
#             # Update the appropriate field based on the interaction type
#             if interaction_type == 'general_query':
#                 update_query = """
#                     UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'feedback':
#                 update_query = """
#                     UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'order_related_issue':
#                 update_query = """
#                     UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
#                 """
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})

#             cursor.execute(update_query, (user_input, interaction_id))

#         else:
#             # Insert a new row if no partially filled row exists
#             insert_query = """
#                 INSERT INTO chatbot (general_queries, feedback, order_related_issues)
#                 VALUES (%s, %s, %s)
#             """
#             # Initialize all values to None, update only the relevant field
#             if interaction_type == 'general_query':
#                 cursor.execute(insert_query, (user_input, None, None))
#             elif interaction_type == 'feedback':
#                 cursor.execute(insert_query, (None, user_input, None))
#             elif interaction_type == 'order_related_issue':
#                 cursor.execute(insert_query, (None, None, user_input))
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()

#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message
    
#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Thank you. Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)
        
#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
    
#     return jsonify({"response": res})


# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)






# #30/08/24
# #database

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\food\\myenv\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }



# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn


# # # Function to save user interaction in the database
# # def save_interaction(interaction_type, user_input):
# #     try:
# #         conn = get_db_connection()
# #         cursor = conn.cursor()
# #         # Adjusted to match the actual interaction types
# #         if interaction_type == 'general_query':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (general_queries)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'feedback':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (feedback)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         elif interaction_type == 'order_related_issue':
# #             insert_query = sql.SQL("""
# #                 INSERT INTO chatbot (order_related_issues)
# #                 VALUES (%s)
# #             """)
# #             cursor.execute(insert_query, (user_input,))
# #         else:
# #             # Handle unknown interaction types if necessary
# #             print(f"Unknown interaction type: {interaction_type}")
# #             return jsonify({"response": "An unknown error occurred. Please try again later."})
        
# #         conn.commit()
# #     except Exception as e:
# #         print(f"Error saving interaction: {e}")
# #         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
# #     finally:
# #         cursor.close()
# #         conn.close()


# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Check if there is an existing row that needs to be updated
#         select_query = """
#             SELECT interaction_id FROM chatbot 
#             WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
#             ORDER BY interaction_id DESC LIMIT 1
#         """
#         cursor.execute(select_query)
#         result = cursor.fetchone()
        
#         # Determine if we should insert a new row or update an existing one
#         if result:
#             interaction_id = result[0]
#             # Update the appropriate field based on the interaction type
#             if interaction_type == 'general_query':
#                 update_query = """
#                     UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'feedback':
#                 update_query = """
#                     UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'order_related_issue':
#                 update_query = """
#                     UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
#                 """
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})

#             cursor.execute(update_query, (user_input, interaction_id))

#         else:
#             # Insert a new row if no partially filled row exists
#             insert_query = """
#                 INSERT INTO chatbot (general_queries, feedback, order_related_issues)
#                 VALUES (%s, %s, %s)
#             """
#             # Initialize all values to None, update only the relevant field
#             if interaction_type == 'general_query':
#                 cursor.execute(insert_query, (user_input, None, None))
#             elif interaction_type == 'feedback':
#                 cursor.execute(insert_query, (None, user_input, None))
#             elif interaction_type == 'order_related_issue':
#                 cursor.execute(insert_query, (None, None, user_input))
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()

#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message
    
#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Thank you. Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)
        
#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
    
#     return jsonify({"response": res})


# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)












# #02/09/2024
# #trying track current order 
# #finaly we did track current order on 9/9/24

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql
# import time

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\Users\\Viswajith\\Desktop\\chat-3\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }



# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# track_order_mode = False  # New mode for tracking orders
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn

# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Check if there is an existing row that needs to be updated
#         select_query = """
#             SELECT interaction_id FROM chatbot 
#             WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
#             ORDER BY interaction_id DESC LIMIT 1
#         """
#         cursor.execute(select_query)
#         result = cursor.fetchone()
        
#         # Determine if we should insert a new row or update an existing one
#         if result:
#             interaction_id = result[0]
#             # Update the appropriate field based on the interaction type
#             if interaction_type == 'general_query':
#                 update_query = """
#                     UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'feedback':
#                 update_query = """
#                     UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'order_related_issue':
#                 update_query = """
#                     UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
#                 """
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})

#             cursor.execute(update_query, (user_input, interaction_id))

#         else:
#             # Insert a new row if no partially filled row exists
#             insert_query = """
#                 INSERT INTO chatbot (general_queries, feedback, order_related_issues)
#                 VALUES (%s, %s, %s)
#             """
#             # Initialize all values to None, update only the relevant field
#             if interaction_type == 'general_query':
#                 cursor.execute(insert_query, (user_input, None, None))
#             elif interaction_type == 'feedback':
#                 cursor.execute(insert_query, (None, user_input, None))
#             elif interaction_type == 'order_related_issue':
#                 cursor.execute(insert_query, (None, None, user_input))
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()

#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()

# def track_order(order_number):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Query to get the order status
#         select_query = """
#             SELECT status, estimated_delivery_time, reason_for_delay FROM user_order
#             WHERE order_number = %s
#         """
#         cursor.execute(select_query, (order_number,))
#         result = cursor.fetchone()

#         if result:
#             time.sleep(3)
            
#             status, estimated_delivery_time, reason_for_delay = result
#             if status == "being prepared":
#                 response = "Your order is currently being prepared. It will be ready for delivery shortly. We'll notify you once it's out for delivery."
#             elif status == "out for delivery":
#                 response = f"Your order is on the way! You can expect it to arrive in approximately {estimated_delivery_time}. If you have any issues, please contact our delivery team at +91 73 7005 7005 ."
#             elif status == "delayed":
#                 response = f"We apologize for the delay. Your order is on the way but is running late due to {reason_for_delay}. The new estimated delivery time is {estimated_delivery_time}. Thank you for your patience."
#             elif status == "delivered":
#                 response = "Your order has been delivered. We hope you enjoy your meal! If you have any feedback or issues, please let us know."
#             else:
#                 response = "Unknown order status. Please contact support for more details."
#         else:
#             response = "Order number not found. Please check and provide a valid order number."

#     except Exception as e:
#         print(f"Error tracking order: {e}")
#         response = "An error occurred while tracking your order. Please try again later."
#     finally:
#         cursor.close()
#         conn.close()

#     return response



# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, track_order_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message

#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif track_order_mode:
#         if message.isdigit():
#             order_number = message
#             res = track_order(order_number)
#             track_order_mode = False  # Resetting the mode after tracking the order
#             order_number = None  # Reset the order number
#         else:
#             res = "You entered an invalid order number. Please provide a valid numeric order number."
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)

#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
#         elif ints and ints[0]['intent'] == 'track_current_order':
#             track_order_mode = True
#             res = "Please provide your order number to track the status."

#     return jsonify({"response": res})



# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)













# # #10/09/2024
# # # checking offer and combos change every week

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql
# from datetime import datetime
# import datetime
# import time

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\Users\\Viswajith\\Desktop\\chat-3\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "perungudi": {
#         "address": "49, 50 Rajeshwari Street, Santhosh Nagar, Perungudi, Chennai, Tamil Nadu 600041.",
#         "contact": "091-50257666",
#         "coords": (12.9592, 80.2446)
#     },
#     "uthandi": {
#         "address": "No 402/203, VGP Golden Beach Layout, East Coast Rd, Uthandi, Chennai, Tamil Nadu 600119.",
#         "contact": "044-29522952",
#         "coords": (12.8693, 80.2435)
#     },
#     "mahindra_city": {
#         "address": "Mahindra City Veerapuram Village, Mahindra World City, Kancheepuram, Tamil Nadu 603002.",
#         "contact": "072-00387493",
#         "coords": (12.7369, 80.0144)
#     },
#     "mylapore": {
#         "address": "Ganesh Arcades, 13/1, Chandrabagh Ave 2nd St, Jagadambal Colony, Othavadi, Mylapore, Chennai, Tamil Nadu 600004.",
#         "contact": "091-50257666",
#         "coords": (13.0368, 80.2676)
#     },
#     "vivira_mall": {
#         "address": "4th floor, Vivira Mall, OMR Rd, Navalur, Tamil Nadu 600130.",
#         "contact": "073-70057005",
#         "coords": (12.8504, 80.2261)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),
#     "nungambakkam": (13.0569, 80.24250),
#     "adyar": (13.0012, 80.2565),
#     "velachery": (12.9755, 80.2207),
#     "thiruvanmiyur": (12.9830, 80.2594),
#     "t_nagar": (13.0418, 80.2341),
#     "guindy": (13.0067, 80.2206),
#     "egmore": (13.0732, 80.2609),
#     "kodambakkam": (13.0521, 80.2255),
#     "besant_nagar": (13.0003, 80.2667),
#     "tambaram": (12.9249, 80.1000),
#     "tharamani": (12.9863, 80.2432),
#     "sholinganallur": (12.9010, 80.2279),
#     "anna_nagar": (13.0850, 80.2101),
#     "porur": (13.0382, 80.1565),
#     "pallavaram": (12.9675, 80.1491),
#     "chromepet": (12.9516, 80.1462),
#     "medavakkam": (12.9200, 80.1920),
#     "madipakkam": (12.9647, 80.1961),
#     "saidapet": (13.0213, 80.2231),
#     "navalur": (12.8459, 80.2265),
#     "teynampet": (13.0405, 80.2503),
#     "thousand_lights": (13.0617, 80.2544),
#     "chetpet": (13.0714, 80.2417),
#     "alandur": (12.9975, 80.2006),
#     "adambakkam": (12.9880, 80.2047),
#     "triplicane": (13.0588, 80.2756),
#     "nanganallur": (12.9807, 80.1882),
#     "pallikaranai": (12.9349, 80.2137),
#     "keelkattalai": (12.9556, 80.1869),
#     "kovilambakkam": (12.9409, 80.1851),
#     "thoraipakkam": (12.9416, 80.2362),
#     "neelankarai": (12.9492, 80.2547),
#     "injambakkam": (12.9198, 80.2511),
#     "hastinapuram": (17.3168, 78.5513),
#     "pozhichur": (12.9898, 80.1434),
#     "pammal": (12.9749, 80.1328),
#     "nagalkeni": (12.9646, 80.1359),
#     "selaiyur": (12.9068, 80.1425),
#     "irumbuliyur": (12.9172, 80.1077),
#     "kadaperi": (12.9336, 80.1254),
#     "perungalathur": (12.9049, 80.0846),
#     "pazhavanthangal": (12.9890, 80.1882),
#     "peerkankaranai": (12.9093, 80.1024),
#     "mudichur": (12.9150, 80.0720),
#     "vandalur": (12.8913, 80.0810),
#     "kolappakkam": (12.7897, 80.2216),
#     "mambakkam": (12.8385, 80.1697),
#     "palavakkam": (12.9617, 80.2562),
#     "varadharajapuram": (12.9266, 80.0758),
#     "west_mambalam": (13.0383, 80.2209),
#     "kottivakkam": (12.9682, 80.2599),
#     "pudupet": (13.0691, 80.2642),
#     "porur": (13.0382, 80.1565),
#     "kovur": (14.5021, 79.9853),
#     "aminjikarai": (13.0698, 80.2245),
#     "ayanavaram": (13.0986, 80.2337),
#     "ambattur": (13.1186, 80.1574),
#     "kundrathur": (12.9977, 80.0972),
#     "mannurpet": (13.0992, 80.1756),
#     "padi": (13.0965, 80.1845),
#     "ayappakkam": (13.0983, 80.1367),
#     "korattur": (13.1082, 80.1834),
#     "mogappair": (13.0837, 80.1750),
#     "arumbakkam": (13.0724, 80.2102),
#     "avadi": (13.1067, 80.0970),
#     "pudur": (13.1299, 80.1603),
#     "maduravoyal": (13.0656, 80.1608),
#     "koyambedu": (13.0694, 80.1948),
#     "ashok_nagar": (13.0373, 80.2123),
#     "k_k_nagar": (13.0410, 80.1994),
#     "karambakkam": (13.0376, 80.1532),
#     "vadapalani": (13.0500, 80.2121),
#     "saligramam": (13.0545, 80.2011),
#     "virugambakkam": (13.0532, 80.1922),
#     "alwarthirunagar": (13.0426, 80.1840),
#     "valasaravakkam": (13.0403, 80.1723),
#     "thirunindravur": (13.1181, 80.0336),
#     "pattabiram": (13.1231, 80.0593),
#     "thirumangalam": (9.8216, 77.9891),
#     "thirumullaivayal": (13.1307, 80.1314),
#     "thiruverkadu": (13.0734, 80.1269),
#     "nandambakkam": (12.9824, 80.0603),
#     "nerkundrum": (13.0678, 80.1859),
#     "nesapakkam": (13.0379, 80.1920),
#     "nolambur": (13.0754, 80.1680),
#     "ramapuram": (13.0317, 80.1817),
#     "mugalivakkam": (13.0210, 80.1614),
#     "mangadu": (13.0270, 80.1107),
#     "m_g_r_nagar": (13.0352, 80.1973),
#     "alapakkam": (13.0490, 80.1673),
#     "poonamallee": (13.0473, 80.0945),
#     "mowlivakkam": (13.0215, 80.1443),
#     "gerugambakkam": (13.0136, 80.1353),
#     "thirumazhisai": (13.0525, 80.0603),
#     "iyyapanthangal": (13.0381, 80.1354),
#     "sikkarayapuram": (13.0150, 80.1030),
#     "red_hills": (13.1865, 80.1999),  
#     "royapuram": (13.1137, 80.2954),   
#     "korukkupet": (13.1186, 80.2780),  
#     "vyasarpadi": (13.1184, 80.2594),  
#     "perambur": (13.1210, 80.2326),    
#     "tondiarpet": (13.1261, 80.2880),  
#     "tiruvottiyur": (13.1643, 80.3001),
#     "ennore": (13.2146, 80.3203),     
#     "minjur": (13.2789, 80.2623),     
#     "old_washermenpet": (13.1148, 80.2872), 
#     "madhavaram": (13.1488, 80.2306),  
#     "manali_new_town": (13.1933, 80.2708), 
#     "naravarikuppam": (13.1670, 80.1929), 
#     "puzhal": (13.1585, 80.2037),    
#     "moolakadai": (13.1296, 80.2416), 
#     "kodungaiyur": (13.1375, 80.2478), 
#     "madhavaram_milk_colony": (13.1505, 80.2419), 
#     "surapet": (13.1454, 80.1838),    
#     "parrys_corner": (13.0896, 80.2882), 
#     "vallalar_nagar": (13.1328, 80.1761), 
#     "new_washermenpet": (13.1148, 80.2872), 
#     "mannadi": (13.0938, 80.2891),     
#     "basin_bridge": (13.1029, 80.2712), 
#     "park_town": (13.0796, 80.2752),  
#     "periamet": (13.0829, 80.2660),    
#     "pattalam": (13.1001, 80.2615),    
#     "pulianthope": (13.0982, 80.2683), 
#     "mkb_nagar": (13.1258, 80.2622),   
#     "selavoyal": (13.1442, 80.2556),   
#     "manjambakkam": (13.1551, 80.2224), 
#     "ponniammanmedu": (13.1350, 80.2274), 
#     "sembiam": (13.1154, 80.2367),    
#     "tvk_nagar": (13.1199, 80.2342),  
#     "icf_colony": (13.0981, 80.2195), 
#     "villivakkam": (13.1086, 80.2061), 
#     "kathivakkam": (13.2161, 80.3182),
#     "kathirvedu": (13.1521, 80.2001),  
#     "erukanchery": (13.1272, 80.2534), 
#     "broadway": (13.0877, 80.2839),  
#     "jamalia": (13.1048, 80.2533),     
#     "kosapet": (13.0922, 80.2551),    
#     "otteri": (13.0921, 80.2510),             
#     "radha_nagar": (12.9535, 80.1444)  
# }


# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# track_order_mode = False  # New mode for tracking orders
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# # def get_response(intents_list, intents_json):
# #     if not intents_list:
# #         return "Sorry, I didn't understand that. Can you please rephrase?"
    
# #     tag = intents_list[0]['intent']
# #     list_of_intents = intents_json['intents']
# #     for i in list_of_intents:
# #         if i['tag'] == tag:
# #             return random.choice(i['responses'])
# #     return "Sorry, I didn't understand that. Can you please rephrase?"

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn

# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Check if there is an existing row that needs to be updated
#         select_query = """
#             SELECT interaction_id FROM chatbot 
#             WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
#             ORDER BY interaction_id DESC LIMIT 1
#         """
#         cursor.execute(select_query)
#         result = cursor.fetchone()
        
#         # Determine if we should insert a new row or update an existing one
#         if result:
#             interaction_id = result[0]
#             # Update the appropriate field based on the interaction type
#             if interaction_type == 'general_query':
#                 update_query = """
#                     UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'feedback':
#                 update_query = """
#                     UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'order_related_issue':
#                 update_query = """
#                     UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
#                 """
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})

#             cursor.execute(update_query, (user_input, interaction_id))

#         else:
#             # Insert a new row if no partially filled row exists
#             insert_query = """
#                 INSERT INTO chatbot (general_queries, feedback, order_related_issues)
#                 VALUES (%s, %s, %s)
#             """
#             # Initialize all values to None, update only the relevant field
#             if interaction_type == 'general_query':
#                 cursor.execute(insert_query, (user_input, None, None))
#             elif interaction_type == 'feedback':
#                 cursor.execute(insert_query, (None, user_input, None))
#             elif interaction_type == 'order_related_issue':
#                 cursor.execute(insert_query, (None, None, user_input))
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()

#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()

# def track_order(order_number):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Query to get the order status
#         select_query = """
#             SELECT status, estimated_delivery_time, reason_for_delay FROM user_order
#             WHERE order_number = %s
#         """
#         cursor.execute(select_query, (order_number,))
#         result = cursor.fetchone()

#         if result:
#             time.sleep(4)  # 2 seconds delay, you can adjust this as needed
            
#             status, estimated_delivery_time, reason_for_delay = result
#             if status == "being prepared":
#                 response = "Your order is currently being prepared. It will be ready for delivery shortly. We'll notify you once it's out for delivery."
#             elif status == "out for delivery":
#                 response = f"Your order is on the way! You can expect it to arrive in approximately {estimated_delivery_time}. If you have any issues, please contact our delivery team at +91 73 7005 7005 ."
#             elif status == "delayed":
#                 response = f"We apologize for the delay. Your order is on the way but is running late due to {reason_for_delay}. The new estimated delivery time is {estimated_delivery_time}. Thank you for your patience."
#             elif status == "delivered":
#                 response = "Your order has been delivered. We hope you enjoy your meal! If you have any feedback or issues, please let us know."
#             else:
#                 response = "Unknown order status. Please contact support for more details."
#         else:
#             response = "Order number not found. Please check and provide a valid order number."

#     except Exception as e:
#         print(f"Error tracking order: {e}")
#         response = "An error occurred while tracking your order. Please try again later."
#     finally:
#         cursor.close()
#         conn.close()

#     return response

# def get_offers_from_db():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         today = datetime.date.today()  
#         select_query = """
#             SELECT offer FROM offers
#             WHERE start_date <= %s AND end_date >= %s AND is_active = TRUE
#         """
#         cursor.execute(select_query, (today, today))
#         offers = cursor.fetchall()
#         return offers
#     except Exception as e:
#         print(f"Error fetching offers: {e}")
#         return []
#     finally:
#         cursor.close()
#         conn.close()

# # def get_response(intents_list, intents_json):
# #     if not intents_list:
# #         return "Sorry, I didn't understand that. Can you please rephrase?"
    
# #     tag = intents_list[0]['intent']
# #     list_of_intents = intents_json['intents']
# #     for i in list_of_intents:
# #         if i['tag'] == tag:
# #             if tag == 'view_offers_and_combos':
# #                 offers = get_offers_from_db()
# #                 offer_buttons = "".join([f"<button class='deal-button' onclick=\"addToCart('{offer[0]}')\">{offer[0]}</button><br>" for offer in offers])
# #                 return f"Here are our current offers and combos:<br><br>{offer_buttons}"
# #             return random.choice(i['responses'])
# #     return "Sorry, I didn't understand that. Can you please rephrase?"

# #added no offer and combos statement
# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"

#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             if tag == 'view_offers_and_combos':
#                 offers = get_offers_from_db()
#                 if not offers:  # Check if the offers list is empty
#                     return "Currently, there are no offers or combos available. Please check back later!"
                
#                 offer_buttons = "".join([f"<button class='deal-button' onclick=\"addToCart('{offer[0]}')\">{offer[0]}</button><br>" for offer in offers])
#                 return f"Here are our current offers and combos:<br><br>{offer_buttons}"
            
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, track_order_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message

#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif track_order_mode:
#         if message.isdigit():
#             order_number = message
#             res = track_order(order_number)
#             track_order_mode = False  # Resetting the mode after tracking the order
#             order_number = None  # Reset the order number
#         else:
#             res = "You entered an invalid order number. Please provide a valid numeric order number."
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)

#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
#         elif ints and ints[0]['intent'] == 'track_current_order':
#             track_order_mode = True
#             res = "Please provide your order number to track the status."

#     return jsonify({"response": res})



# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)












# # # #12/09/2024

# from flask import Flask, request, jsonify, render_template
# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import psycopg2
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from geopy.distance import geodesic
# from flask_sqlalchemy import SQLAlchemy
# from psycopg2 import sql
# from datetime import datetime
# import datetime
# import time

# app = Flask(__name__)

# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open("C:\\Users\\Viswajith\\Desktop\\chat-3\\intents.json").read())

# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbot_food.h5')

# stores = {
#     "gopalapuram": {
#         "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
#         "contact": "044-29522952",
#         "coords": (13.0489, 80.2586)
#     },
#     "chromepet": {
#         "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
#         "contact": "073-70057005",
#         "coords": (12.9516, 80.1462)
#     }
# }

# location_coords = {
#     "santhome": (13.0319, 80.2788),            
#     "radha_nagar": (12.9535, 80.1444)  
# }


# cart = []  # In-memory cart

# # Initialize mode variables
# feedback_mode = False
# general_query_mode = False
# store_near_me_mode = False
# order_related_mode = False
# track_order_mode = False  # New mode for tracking orders
# order_number = None

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, word in enumerate(words):
#             if word == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

# def find_nearest_store(location):
#     location = location.lower().replace(" ", "_")
    
#     if location in stores:
#         store_info = stores[location]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     elif location in location_coords:
#         user_coords = location_coords[location]
#         nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#         store_info = nearest_store[1]
#         return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#     else:
#         return "Sorry, we couldn't find any stores near your location."

# # Database connection function
# def get_db_connection():
#     conn = psycopg2.connect(
#         host="localhost",  
#         dbname="chat-1",  
#         user="postgres",  
#         password="@Viswa2000"  
#     )
#     return conn

# # Function to save user interaction in the database
# def save_interaction(interaction_type, user_input):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Check if there is an existing row that needs to be updated
#         select_query = """
#             SELECT interaction_id FROM chatbot 
#             WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
#             ORDER BY interaction_id DESC LIMIT 1
#         """
#         cursor.execute(select_query)
#         result = cursor.fetchone()
        
#         # Determine if we should insert a new row or update an existing one
#         if result:
#             interaction_id = result[0]
#             # Update the appropriate field based on the interaction type
#             if interaction_type == 'general_query':
#                 update_query = """
#                     UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'feedback':
#                 update_query = """
#                     UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
#                 """
#             elif interaction_type == 'order_related_issue':
#                 update_query = """
#                     UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
#                 """
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})

#             cursor.execute(update_query, (user_input, interaction_id))

#         else:
#             # Insert a new row if no partially filled row exists
#             insert_query = """
#                 INSERT INTO chatbot (general_queries, feedback, order_related_issues)
#                 VALUES (%s, %s, %s)
#             """
#             # Initialize all values to None, update only the relevant field
#             if interaction_type == 'general_query':
#                 cursor.execute(insert_query, (user_input, None, None))
#             elif interaction_type == 'feedback':
#                 cursor.execute(insert_query, (None, user_input, None))
#             elif interaction_type == 'order_related_issue':
#                 cursor.execute(insert_query, (None, None, user_input))
#             else:
#                 print(f"Unknown interaction type: {interaction_type}")
#                 return jsonify({"response": "An unknown error occurred. Please try again later."})
        
#         conn.commit()

#     except Exception as e:
#         print(f"Error saving interaction: {e}")
#         return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
#     finally:
#         cursor.close()
#         conn.close()

# def track_order(order_number):
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         # Query to get the order status
#         select_query = """
#             SELECT status, estimated_delivery_time, reason_for_delay FROM user_order
#             WHERE order_number = %s
#         """
#         cursor.execute(select_query, (order_number,))
#         result = cursor.fetchone()

#         if result:
#             time.sleep(4)  # 2 seconds delay, you can adjust this as needed
            
#             status, estimated_delivery_time, reason_for_delay = result
#             if status == "being prepared":
#                 response = "Your order is currently being prepared. It will be ready for delivery shortly. We'll notify you once it's out for delivery."
#             elif status == "out for delivery":
#                 response = f"Your order is on the way! You can expect it to arrive in approximately {estimated_delivery_time}. If you have any issues, please contact our delivery team at +91 73 7005 7005 ."
#             elif status == "delayed":
#                 response = f"We apologize for the delay. Your order is on the way but is running late due to {reason_for_delay}. The new estimated delivery time is {estimated_delivery_time}. Thank you for your patience."
#             elif status == "delivered":
#                 response = "Your order has been delivered. We hope you enjoy your meal! If you have any feedback or issues, please let us know."
#             else:
#                 response = "Unknown order status. Please contact support for more details."
#         else:
#             response = "Order number not found. Please check and provide a valid order number."

#     except Exception as e:
#         print(f"Error tracking order: {e}")
#         response = "An error occurred while tracking your order. Please try again later."
#     finally:
#         cursor.close()
#         conn.close()

#     return response

# def get_offers_from_db():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         today = datetime.date.today()  
#         select_query = """
#             SELECT offer FROM offers
#             WHERE start_date <= %s AND end_date >= %s AND is_active = TRUE
#         """
#         cursor.execute(select_query, (today, today))
#         offers = cursor.fetchall()
#         return offers
#     except Exception as e:
#         print(f"Error fetching offers: {e}")
#         return []
#     finally:
#         cursor.close()
#         conn.close()

# #added no offer and combos statement
# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I didn't understand that. Can you please rephrase?"

#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             if tag == 'view_offers_and_combos':
#                 offers = get_offers_from_db()
#                 if not offers:  # Check if the offers list is empty
#                     return "Currently, there are no offers or combos available. Please check back later!"
                
#                 offer_buttons = "".join([f"<button class='deal-button' onclick=\"addToCart('{offer[0]}')\">{offer[0]}</button><br>" for offer in offers])
#                 return f"Here are our current offers and combos:<br><br>{offer_buttons}<br>Would you like to add any to your cart?"
            
#             return random.choice(i['responses'])
#     return "Sorry, I didn't understand that. Can you please rephrase?"

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, track_order_mode, order_number

#     message = request.json.get("message").strip().lower()  # Normalize the message

#     if store_near_me_mode:
#         if message in ["enter location", "input location", "type location", "provide location"]:
#             res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
#         else:
#             res = find_nearest_store(message)
#             store_near_me_mode = False  # Resetting the mode after use
#     elif feedback_mode:
#         res = get_response([{'intent': 'feedback_response'}], intents)
#         save_interaction('feedback', message)  # Save interaction
#         feedback_mode = False
#     elif general_query_mode:
#         res = get_response([{'intent': 'general_query_response'}], intents)
#         save_interaction('general_query', message)  # Save interaction
#         general_query_mode = False
#     elif track_order_mode:
#         if message.isdigit():
#             order_number = message
#             res = track_order(order_number)
#             track_order_mode = False  # Resetting the mode after tracking the order
#             order_number = None  # Reset the order number
#         else:
#             res = "You entered an invalid order number. Please provide a valid numeric order number."
#     elif order_related_mode:
#         if order_number is None:
#             if message.isdigit():
#                 order_number = message
#                 res = "Please describe the issue you are facing with your order."
#             else:
#                 res = "You entered an invalid order number. Please provide a valid numeric order number."
#         else:
#             res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
#             save_interaction('order_related_issue', message)  # Save interaction
#             order_related_mode = False  # Resetting the mode after handling the issue
#             order_number = None  # Reset the order number
#     else:
#         ints = predict_class(message)
#         res = get_response(ints, intents)

#         # Check for specific intents to set the mode
#         if ints and ints[0]['intent'] == 'general_queries':
#             general_query_mode = True
#         elif ints and ints[0]['intent'] == 'feedback':
#             feedback_mode = True
#         elif ints and ints[0]['intent'] == 'store_near_me':
#             store_near_me_mode = True
#         elif ints and ints[0]['intent'] == 'view_offers_and_combos':
#             res = get_response(ints, intents)
#             cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
#         elif ints and ints[0]['intent'] == 'order_related_issues':
#             order_related_mode = True
#             res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
#         elif ints and ints[0]['intent'] == 'track_current_order':
#             track_order_mode = True
#             res = "Please provide your order number to track the status."

#     return jsonify({"response": res})



# @app.route("/share_location", methods=["POST"])
# def share_location():
#     global store_near_me_mode

#     data = request.json
#     permission = data.get("permission")  # The user's choice for location sharing
#     latitude = data.get("latitude")
#     longitude = data.get("longitude")

#     if permission == "allow_once":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         store_near_me_mode = False  # Reset the mode after use

#     elif permission == "allow_all_time":
#         if latitude is not None and longitude is not None:
#             user_coords = (latitude, longitude)
#             nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
#             store_info = nearest_store[1]
#             response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
#         else:
#             response = "Unable to determine your location."
#         # Store the user's location for future requests without asking permission again
#         store_near_me_mode = True  # Keep the mode active for future use

#     elif permission == "deny":
#         response = "You have denied location access. Unable to provide store information."

#     else:
#         response = "Invalid permission option."

#     return jsonify({"response": response})


# @app.route("/welcome", methods=["GET"])
# def welcome():
#     welcome_message = get_response([{'intent': 'welcome'}], intents)
#     return jsonify({"response": welcome_message})

# @app.route("/add_offer_to_cart", methods=["POST"])
# def add_offer_to_cart():
#     offer = request.json.get("offer")
#     if offer:
#         cart.append(offer)
#         return jsonify({"response": f"'{offer}' has been added to your cart."})
#     return jsonify({"response": "No offer specified."})

# if __name__ == "__main__":
#     app.run(debug=True)
















# # #12/09/2024
#trying to print checking the status of your order.....   finally we did this message on 18/09/2024

from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
import psycopg2
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from geopy.distance import geodesic
from flask_sqlalchemy import SQLAlchemy
from psycopg2 import sql
from datetime import datetime
import datetime
import time

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("C:\\Users\\Viswajith\\Desktop\\chat-3\\intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_food.h5')

stores = {
    "gopalapuram": {
        "address": "40, Cathedral Rd, near Stella Maris College, Gopalapuram, Chennai, Tamil Nadu 600086.",
        "contact": "044-29522952",
        "coords": (13.0489, 80.2586)
    },
    "chromepet": {
        "address": "Old No.32, New No.52, 1, Station Rd, Radha Nagar, Chromepet, Chennai, Tamil Nadu 600044.",
        "contact": "073-70057005",
        "coords": (12.9516, 80.1462)
    }
}

location_coords = {
    "santhome": (13.0319, 80.2788),            
    "radha_nagar": (12.9535, 80.1444)  
}


cart = []  # In-memory cart

# Initialize mode variables
feedback_mode = False
general_query_mode = False
store_near_me_mode = False
order_related_mode = False
track_order_mode = False  # New mode for tracking orders
order_number = None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def find_nearest_store(location):
    location = location.lower().replace(" ", "_")
    
    if location in stores:
        store_info = stores[location]
        return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
    elif location in location_coords:
        user_coords = location_coords[location]
        nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
        store_info = nearest_store[1]
        return f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
    else:
        return "Sorry, we couldn't find any stores near your location."

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(
        host="localhost",  
        dbname="chat-1",  
        user="postgres",  
        password="@Viswa2000"  
    )
    return conn

# Function to save user interaction in the database
def save_interaction(interaction_type, user_input):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if there is an existing row that needs to be updated
        select_query = """
            SELECT interaction_id FROM chatbot 
            WHERE (general_queries IS NULL OR feedback IS NULL OR order_related_issues IS NULL)
            ORDER BY interaction_id DESC LIMIT 1
        """
        cursor.execute(select_query)
        result = cursor.fetchone()
        
        # Determine if we should insert a new row or update an existing one
        if result:
            interaction_id = result[0]
            # Update the appropriate field based on the interaction type
            if interaction_type == 'general_query':
                update_query = """
                    UPDATE chatbot SET general_queries = %s WHERE interaction_id = %s
                """
            elif interaction_type == 'feedback':
                update_query = """
                    UPDATE chatbot SET feedback = %s WHERE interaction_id = %s
                """
            elif interaction_type == 'order_related_issue':
                update_query = """
                    UPDATE chatbot SET order_related_issues = %s WHERE interaction_id = %s
                """
            else:
                print(f"Unknown interaction type: {interaction_type}")
                return jsonify({"response": "An unknown error occurred. Please try again later."})

            cursor.execute(update_query, (user_input, interaction_id))

        else:
            # Insert a new row if no partially filled row exists
            insert_query = """
                INSERT INTO chatbot (general_queries, feedback, order_related_issues)
                VALUES (%s, %s, %s)
            """
            # Initialize all values to None, update only the relevant field
            if interaction_type == 'general_query':
                cursor.execute(insert_query, (user_input, None, None))
            elif interaction_type == 'feedback':
                cursor.execute(insert_query, (None, user_input, None))
            elif interaction_type == 'order_related_issue':
                cursor.execute(insert_query, (None, None, user_input))
            else:
                print(f"Unknown interaction type: {interaction_type}")
                return jsonify({"response": "An unknown error occurred. Please try again later."})
        
        conn.commit()

    except Exception as e:
        print(f"Error saving interaction: {e}")
        return jsonify({"response": "An error occurred while saving your feedback. Please try again later."})
    finally:
        cursor.close()
        conn.close()

def track_order(order_number):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to get the order status
        select_query = """
            SELECT status, estimated_delivery_time, reason_for_delay FROM user_order
            WHERE order_number = %s
        """
        cursor.execute(select_query, (order_number,))
        result = cursor.fetchone()

        if result:
            time.sleep(4)  # 2 seconds delay, you can adjust this as needed
            
            status, estimated_delivery_time, reason_for_delay = result
            if status == "being prepared":
                response = "Your order is currently being prepared. It will be ready for delivery shortly. We'll notify you once it's out for delivery."
            elif status == "out for delivery":
                response = f"Your order is on the way! You can expect it to arrive in approximately {estimated_delivery_time}. If you have any issues, please contact our delivery team at +91 73 7005 7005 ."
            elif status == "delayed":
                response = f"We apologize for the delay. Your order is on the way but is running late due to {reason_for_delay}. The new estimated delivery time is {estimated_delivery_time}. Thank you for your patience."
            elif status == "delivered":
                response = "Your order has been delivered. We hope you enjoy your meal! If you have any feedback or issues, please let us know."
            else:
                response = "Unknown order status. Please contact support for more details."
        else:
            response = "Order number not found. Please check and provide a valid order number."

    except Exception as e:
        print(f"Error tracking order: {e}")
        response = "An error occurred while tracking your order. Please try again later."
    finally:
        cursor.close()
        conn.close()

    return response

def get_offers_from_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        today = datetime.date.today()  
        select_query = """
            SELECT offer FROM offers
            WHERE start_date <= %s AND end_date >= %s AND is_active = TRUE
        """
        cursor.execute(select_query, (today, today))
        offers = cursor.fetchall()
        return offers
    except Exception as e:
        print(f"Error fetching offers: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

#added no offer and combos statement
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == 'view_offers_and_combos':
                offers = get_offers_from_db()
                if not offers:  # Check if the offers list is empty
                    return "Currently, there are no offers or combos available. Please check back later!"
                
                offer_buttons = "".join([f"<button class='deal-button' onclick=\"addToCart('{offer[0]}')\">{offer[0]}</button><br>" for offer in offers])
                return f"Here are our current offers and combos:<br><br>{offer_buttons}<br>Would you like to add any to your cart?"
            
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that. Can you please rephrase?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global general_query_mode, feedback_mode, store_near_me_mode, order_related_mode, track_order_mode, order_number

    message = request.json.get("message").strip().lower()  # Normalize the message

    if store_near_me_mode:
        if message in ["enter location", "input location", "type location", "provide location"]:
            res = "Please provide your location so we can find the nearest store for you. Enter your address or area name below:"
        else:
            res = find_nearest_store(message)
            store_near_me_mode = False  # Resetting the mode after use
    elif feedback_mode:
        res = get_response([{'intent': 'feedback_response'}], intents)
        save_interaction('feedback', message)  # Save interaction
        feedback_mode = False
    elif general_query_mode:
        res = get_response([{'intent': 'general_query_response'}], intents)
        save_interaction('general_query', message)  # Save interaction
        general_query_mode = False
    elif track_order_mode:
        if message.isdigit():
            order_number = message
            res = "Checking the status of your order......"  # First message
            track_order_mode = False  # Resetting the mode after tracking the order
            order_status = track_order(order_number)  # Get order status
            order_number = None  # Reset the order number
            return jsonify({"response": res, "track_order_status": order_status})  # Send both responses
        else:
            res = "You entered an invalid order number. Please provide a valid numeric order number."
    elif order_related_mode:
        if order_number is None:
            if message.isdigit():
                order_number = message
                res = "Please describe the issue you are facing with your order."
            else:
                res = "You entered an invalid order number. Please provide a valid numeric order number."
        else:
            res = "We apologize for any inconvenience caused. Our support team will review your issue and get back to you as soon as possible. If you need immediate assistance, please contact our support team at info@fastapizza.com or call us at +91 7370057005."
            save_interaction('order_related_issue', message)  # Save interaction
            order_related_mode = False  # Resetting the mode after handling the issue
            order_number = None  # Reset the order number
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)

        # Check for specific intents to set the mode
        if ints and ints[0]['intent'] == 'general_queries':
            general_query_mode = True
        elif ints and ints[0]['intent'] == 'feedback':
            feedback_mode = True
        elif ints and ints[0]['intent'] == 'store_near_me':
            store_near_me_mode = True
        elif ints and ints[0]['intent'] == 'view_offers_and_combos':
            res = get_response(ints, intents)
            cart.append(ints[0]['intent'])  # Add the offer/combo to the cart based on intent
        elif ints and ints[0]['intent'] == 'order_related_issues':
            order_related_mode = True
            res = "You have selected 'Order-related issues'. Please provide your order number for us to assist you better."
        elif ints and ints[0]['intent'] == 'track_current_order':
            track_order_mode = True
            res = "Please provide your order number to track the status."

    return jsonify({"response": res})



@app.route("/share_location", methods=["POST"])
def share_location():
    global store_near_me_mode

    data = request.json
    permission = data.get("permission")  # The user's choice for location sharing
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if permission == "allow_once":
        if latitude is not None and longitude is not None:
            user_coords = (latitude, longitude)
            nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
            store_info = nearest_store[1]
            response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
        else:
            response = "Unable to determine your location."
        store_near_me_mode = False  # Reset the mode after use

    elif permission == "allow_all_time":
        if latitude is not None and longitude is not None:
            user_coords = (latitude, longitude)
            nearest_store = min(stores.items(), key=lambda x: geodesic(user_coords, x[1]['coords']).km)
            store_info = nearest_store[1]
            response = f"Nearest store:\nAddress: {store_info['address']}\nContact Number: {store_info['contact']}"
        else:
            response = "Unable to determine your location."
        # Store the user's location for future requests without asking permission again
        store_near_me_mode = True  # Keep the mode active for future use

    elif permission == "deny":
        response = "You have denied location access. Unable to provide store information."

    else:
        response = "Invalid permission option."

    return jsonify({"response": response})


@app.route("/welcome", methods=["GET"])
def welcome():
    welcome_message = get_response([{'intent': 'welcome'}], intents)
    return jsonify({"response": welcome_message})

@app.route("/add_offer_to_cart", methods=["POST"])
def add_offer_to_cart():
    offer = request.json.get("offer")
    if offer:
        cart.append(offer)
        return jsonify({"response": f"'{offer}' has been added to your cart."})
    return jsonify({"response": "No offer specified."})

if __name__ == "__main__":
    app.run(debug=True)













# DROP TABLE IF EXISTS chatbot;

# CREATE TABLE chatbot (
#     general_queries TEXT,
#     feedback TEXT,
#     order_related_issues TEXT
# );


# SELECT * FROM chatbot


# INSERT INTO chatbot (general_queries, feedback, order_related_issues)
# VALUES ('How can I track my order?', 'Excellent service.', 'Delayed delivery for order #5678.');



# INSERT INTO chatbot (general_queries, feedback)
# VALUES ('how can i contact you store', 'i love the pizza');

# INSERT INTO chatbot (general_queries, order_related_issues)
# VALUES('i love the mexiacn roll', 'my order was late')

# ALTER TABLE chatbot ADD COLUMN user_id SERIAL PRIMARY KEY;

# ALTER TABLE chatbot DROP COLUMN user_id;

# ALTER TABLE chatbot ADD COLUMN id SERIAL PRIMARY KEY;

# ALTER TABLE chatbot DROP COLUMN id;

# ALTER TABLE chatbot ADD COLUMN interaction_id SERIAL;

# ALTER TABLE chatbot DROP COLUMN interaction_id;

# DELETE FROM chatbot;





# DROP TABLE IF EXISTS orders;
# CREATE TABLE orders (
#     order_number VARCHAR(255) PRIMARY KEY,
#     status VARCHAR(50),
#     estimated_delivery_time VARCHAR(50),
#     reason_for_delay VARCHAR(255)
# );

# SELECT * FROM orders


# INSERT INTO orders (order_number, status, estimated_delivery_time, reason_for_delay) 
# VALUES ('4', 'being prepared', '34 min', 'No delay');

# INSERT INTO orders (order_number, status, estimated_delivery_time, reason_for_delay) 
# VALUES ('6', 'out for delivery', '5 min', 'No delay');

# INSERT INTO orders (order_number, status, estimated_delivery_time, reason_for_delay) 
# VALUES ('5', 'delayed', '10 min', 'trafic');

# INSERT INTO orders (order_number, status, reason_for_delay) 
# VALUES ('3', 'delivered', 'No delay');

# SELECT status, estimated_delivery_time, reason_for_delay 
# FROM orders 
# WHERE order_number = '1';

# DELETE FROM orders;


# DROP TABLE IF EXISTS orders;
# CREATE TABLE orders (
#     order_id SERIAL PRIMARY KEY,
#     order_number VARCHAR(50) UNIQUE NOT NULL,
#     status VARCHAR(50) NOT NULL,
#     estimated_delivery_time VARCHAR(50),
#     reason_for_delay VARCHAR(255)
# );


# INSERT INTO orders (order_number, status, estimated_delivery_time)
# VALUES ('ORD001', 'Being Prepared', '2024-09-04 12:30 PM');

# INSERT INTO orders (order_number, status, estimated_delivery_time)
# VALUES ('ORD002', 'Out for Delivery', '2024-09-04 01:00 PM');

# INSERT INTO orders (order_number, status, estimated_delivery_time, reason_for_delay)
# VALUES ('ORD003', 'Delayed', '2024-09-04 01:30 PM', 'Heavy traffic in the area');

# INSERT INTO orders (order_number, status, estimated_delivery_time)
# VALUES ('ORD004', 'Delivered', '2024-09-04 12:45 PM');


# DROP TABLE IF EXISTS order_status;
# CREATE TABLE order_status (
#     order_number SERIAL PRIMARY KEY,
#     status VARCHAR(50) NOT NULL,
#     estimated_delivery_time TIMESTAMP,
#     reason_for_delay TEXT
# );


# INSERT INTO order_status (order_number, status, estimated_delivery_time) VALUES
# (1, 'being prepared', CURRENT_TIMESTAMP + INTERVAL '30 minutes'),
# (2, 'out for delivery', CURRENT_TIMESTAMP + INTERVAL '15 minutes'),
# (3, 'delivered', CURRENT_TIMESTAMP - INTERVAL '10 minutes'),
# (4, 'delayed', CURRENT_TIMESTAMP + INTERVAL '45 minutes');

# SELECT status, estimated_delivery_time, reason_for_delay 
# FROM order_status 
# WHERE order_id = '1';

# INSERT INTO order_status (status, estimated_delivery_time) VALUES
# ('being prepared', CURRENT_TIMESTAMP + INTERVAL '30 minutes'),
# ('out for delivery', CURRENT_TIMESTAMP + INTERVAL '15 minutes'),
# ('delivered', CURRENT_TIMESTAMP - INTERVAL '10 minutes'),
# ('delayed', CURRENT_TIMESTAMP + INTERVAL '45 minutes');

# select * from order_status

# alter table order_status drop column created_at;

# delete from order_status



# CREATE TABLE order_phone (
#   phone_number TEXT PRIMARY KEY,
#   order_status TEXT,
#   estimated_delivery_time TEXT,
#   reason_for_delay TEXT
# );

# select * from order_phone

# INSERT INTO order_phone (phone_number, order_status, estimated_delivery_time, reason_for_delay) 
# VALUES 
# ('1234567890', 'being prepared', '2024-09-12 15:00', NULL),
# ('9876543210', 'out for delivery', '2024-09-08 12:00', NULL),
# ('5551234567', 'Delayed', '2024-09-10 18:00', 'Weather conditions'),
# ('4445556666', 'delivered', '2024-09-13 09:00', NULL);



# drop table if exists offers;
# CREATE TABLE offers (
#     id SERIAL PRIMARY KEY,
#     offer TEXT,
#     description TEXT,
#     start_date DATE,
#     end_date DATE
# );

# alter table offers drop column description;
# select * from offers

# INSERT INTO offers (offer, start_date, end_date) VALUES
# ('10% Off on Pizza', '2024-09-10', '2024-09-11'),
# ('Buy 1 Get 1 Free', '2024-09-10', '2024-10-11'),
# ('Free Dessert with Orders Above $20', '2024-09-10', '2024-10-11'),
# ('Family Combo Deal', '2024-09-11', '2024-09-12'),
# ('Weekend Special - 15% Off', '2024-09-11', '2024-09-12');

# INSERT INTO offers (offer, start_date, end_date, is_active) VALUES
# ('10% Off on Pizza', '2024-09-10', '2024-09-11', TRUE),
# ('Buy 1 Get 1 Free', '2024-09-10', '2024-10-11', TRUE),
# ('Free Dessert with Orders Above $20', '2024-09-10', '2024-10-11', TRUE),
# ('Family Combo Deal', '2024-09-11', '2024-09-12', TRUE),
# ('Weekend Special - 15% Off', '2024-09-11', '2024-09-12', TRUE);


# UPDATE offers
# SET start_date = '2024-09-10',
#     end_date = '2024-09-10'
# WHERE offer = 'Weekend Special - 15% Off';

# ALTER TABLE offers
# ADD COLUMN is_active BOOLEAN DEFAULT TRUE;


# delete from offers


# select * from offers

# INSERT INTO offers (offer, start_date, end_date, is_active) VALUES
# ('10% Off on Pizza', '2024-09-10', '2024-09-11', TRUE),
# ('Buy 1 Get 1 Free', '2024-09-10', '2024-09-11', TRUE),
# ('Free Dessert with Orders Above $20', '2024-09-10', '2024-10-11', TRUE),
# ('Family Combo Deal', '2024-09-11', '2024-09-12', TRUE),
# ('Weekend Special - 15% Off', '2024-09-11', '2024-09-11', TRUE);

# insert into offers (offer, start_date, end_date, is_active)
# values ('buy one mexican roll free for any odder', '2024-09-12', '2024-09-12', 'true');

# UPDATE offers
# SET start_date = '2024-09-11',
#     end_date = '2024-09-11'
# WHERE offer = 'Weekend Special - 15% Off';

# UPDATE offers
# SET start_date = '2024-09-10',
#     end_date = '2024-09-11'
# WHERE offer = 'buy 1 get 1 free';


# delete from offers
