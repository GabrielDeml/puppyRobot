from a_star import AStar as RobotControl
import argparse
import os 
import openai
import json
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = '''
Commands:
move. set motor [left] [right] -100 to 100
print. print text [text]
wait. wait for [seconds] 0 to 10
led. togle led [red] [yellow] [green] 0 or 1

Respond with a json in the format of the following example:
{
    "commands": [
        {"command": "led", "value": [1,1,1]},
        {"command": "move", "value":  [100,100]},
        {"command": "print", "value":  "text"},
        {"command": "wait", "value":  1}
    ]
}

Each command will be executed sequentially from the response list. Not all commands need to be used. Make sure to stop all motors at the end of the response.

Task:
Draw a square and print the location that you will be going to

Response:
'''




robotControl = RobotControl()
def main():
    # Read imput arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--turnOnLEDs", help="Turn on LED", action="store_true")
    parser.add_argument("--turnOffLEDs", help="Turn off LED", action="store_true")


    args = parser.parse_args()

    if args.turnOnLEDs:
        robotControl.leds(1,1,1)
    elif args.turnOffLEDs:
        robotControl.leds(0,0,0)
    
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=500)
    with open('output.json', 'w') as f:
        f.write(str(response))
    # Parse response with json library
    print(response['choices'][0]['text']) 
    response = json.loads(response['choices'][0]['text'])
    # Execute commands
    for command in response['commands']:
        print(command)
        if command['command'] == 'move':
            robotControl.motors(command['value'][0], command['value'][1])
        elif command['command'] == 'print':
            print(command['value'])
        elif command['command'] == 'wait':
            time.sleep(command['value'])
        elif command['command'] == 'led':
            robotControl.leds(command['value'][0], command['value'][1], command['value'][2])


if __name__ == "__main__":
    main()


