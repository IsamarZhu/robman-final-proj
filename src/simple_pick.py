import argparse
from setup import SimulationEnvironment
from state_machine.state_machine import CafeStateMachine

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="specific pick and place simulation")
    parser.add_argument("scenario", choices=["one", "two", "three"], default="one")
    args = parser.parse_args()

    env = SimulationEnvironment(args.scenario, wsg_open=0.107)
    env.build()

    env.meshcat.StartRecording()
    state_machine = CafeStateMachine(env)
    state_machine.run()
    env.meshcat.StopRecording()
    env.meshcat.PublishRecording()


if __name__ == "__main__":
    main()
