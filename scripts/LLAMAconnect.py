import os
import json

from groq import Groq
from tqdm import tqdm

client = Groq(
    api_key = os.getenv("GROQ_API_KEY"),
)


def getResponse(message):
    # Returns a regular LLAMA response
    instructions_content = (f"{message}"
                            )
    message = [
        {
            "role": "user",
            "content": instructions_content,
        }
    ]

    chat_completion = client.chat.completions.create(messages=message, model="llama3-8b-8192")

    return chat_completion.choices[0].message.content

def getResponse_trip(places):
    # Generate individual responses for the trips

    instructions_content = (f"Plan the information of a regular student." 
                            )
    message = [
        {
            "role": "system",
            "content": "You need to plan the routine of a student who has access to the following places: {places}. I want to know what they do during the day, including having lunch, studying, leisure, shopping and practicing sports. Whenever students are having classes or studying, they must be somewhere where it's possible to study. Whenever students are having lunch, breakfast or dinner, they MUST be at a place that provides food. Whenever students are shopping, they MUST be at somewhere from the allowed places where shopping is possible. Whenever students are praticing sports, they MUST be somewhere where it is possible to practice sports. Do not use any location that has not been provided and WRITE THE LOCATIONS EXACTLY AS PROVIDED. Students have lunch between 12 and 14 and dinner between 18 and 21. Students do not study after 20. Students may only go to the gym once every day. Students spend most of their day having classes. Your response should be in a JSON format showing only the current location and current activity. Never include locations to activities. Always start the day at time 7 at home and end the day at time 23 at home, update every hour. Locations always have a single place, never two. Follow the example where the first number is the time: {{'7': {{'location':'home', 'activity':'wake up'}}, '8': {{'location':'school', 'activity':'study'}}, '9':{{'location':'cafe', 'activity':'breakfast'}}}}.".format(
                places=", ".join(places),
            )
        },
        {
            "role": "user",
            "content": instructions_content,
        }
    ]
    chat_completion = client.chat.completions.create(messages=message, model="llama3-8b-8192", temperature=1, response_format={"type": "json_object"})

    return chat_completion.choices[0].message.content

def responseCheck(response, places):
    # Checks if the response from the LLM makes sense
    response = json.loads(response)
    if len(response) < 17:
        print("Error: The response is missing some hours")
        return False
    for item in response:
        try:
            local = response[item]['location']
            if local not in places or ',' in local:
                print(f"Error: Invalid location generated '{local}'")
                return False
        except KeyError:
            print("Error: The response is missing a key")
            return False
    return True
