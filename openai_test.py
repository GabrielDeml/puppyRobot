import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = '''
Commands:
1. move to position [x,y] with positions [0,0] to [100,100]
2. print text [text]

Respond with a json in the format of the following example:
{
    commands: [
        "1": [1,1],
        "2":'text"
    ]
}

Each command will be executed sequentially from the response list

Task:
Draw a square and print the location that you will be going to
'''


response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=500)
# Write to file
with open('output.json', 'w') as f:
    f.write(str(response))
print(response['choices'][0]['text']) 