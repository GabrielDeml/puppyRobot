from a_star import AStar as RobotControl
import argparse

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
    

if __name__ == "__main__":
    main()


