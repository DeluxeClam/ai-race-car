import pygame
import sys
import random
import math

from settings import *
from car import Car, PlayerCar

import neat
import visualize

# Define game class
class Game:
    def __init__(self) -> None:
        
        # General setup
        pygame.init()
        pygame.display.set_caption('Self driving car school')
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(BACKGROUND_COLOUR)
            pygame.display.update()
            self.clock.tick(FPS)


# Define game class
class TestGame:
    def __init__(self) -> None:
        
        # General setup
        pygame.init()
        pygame.display.set_caption('Self driving car school')
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.player = PlayerCar((700, 610))

        self.game_map = pygame.image.load('map1.png').convert() # Convert Speeds Up A Lot

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(BACKGROUND_COLOUR)
            self.clock.tick(FPS)

            self.player.update(self.game_map)

            self.screen.blit(self.game_map, (0, 0))
            if self.player.is_alive():
                self.player.draw(self.screen)
            
            pygame.display.update()

            self.clock.tick(FPS)



current_generation = 0 # Generation counter
winner = None

def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []
    node_names = {-5:"left", -4:"left-forward", -3:"forward", -2:"right-forward", -1:"right",
                  0:"turn-left", 1:"turn-right", 2:"decelerate", 3:"accelerate"}

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car((700, 610)))

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map1.png').convert() # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.turn(10) # Left
            elif choice == 1:
                car.turn(-10) # Right
            elif choice == 2:
                car.accelerate(-2, min_speed=12)
            else:
                car.accelerate(2) # Speed Up
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40: # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (250, 410)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (250, 450)
        screen.blit(text, text_rect)
        
        if current_generation > 1:
            viz = pygame.image.load('Digraph.gv.png').convert()
            viz_rect = viz.get_rect()
            viz_rect.topleft = (1000, 0)
            screen.blit(viz, viz_rect)

        pygame.display.flip()
        clock.tick(FPS) # 60 FPS
    
    genome_fitness = 0
    genome_index = 0

    for i, g in enumerate(genomes):
        if g[1].fitness > genome_fitness:
            genome_fitness = g[1].fitness
            genome_index = i - 1
    winner = genomes[genome_index][1]
    visualize.draw_net(config, winner, False, node_names=node_names, fmt='png')
    




if __name__ == "__main__":
    
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 1000)

