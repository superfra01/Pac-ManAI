import pickle
import sys
import pygame
from pygame.locals import *
import neat
import os
import copy

import visualize
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
from mazedata import MazeData
import multiprocessing

# Dizionari per interpretare i 4 output della rete neurale:
# - 0 => forward
# - 1 => backward
# - 2 => right
# - 3 => left
#Mappatura output relativi alla direzine del PacMan IA
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, STOP: STOP}
TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP, STOP: STOP}
TURN_LEFT  = {UP: LEFT,  LEFT: DOWN, DOWN: RIGHT, RIGHT: UP, STOP: STOP}

class GameController(object):
    """
    Gioco Pac-Man standard con NEAT.
    (codice base, invariato a parte eventuali piccole modifiche).
    """
    def __init__(self, train_mode=False, net=None, config=None, headless=False, fixed_dt=1.0/60.0):
        self.headless = headless
        self.fixed_dt = fixed_dt
        self.train_mode = train_mode
        self.net = net
        self.neat_config = config
        self.game_over = False
        self.fitness = 0.0

        # Penalità se Pac-Man non mangia pellet per troppo tempo
        self.timeSinceLastPellet = 0.0
        self.idlePelletThreshold = 15.0
        self.idlePenalty = 100

        # Reward se riduciamo la distanza BFS dal pellet più vicino
        self.prevDistanceToClosestPellet = None
        self.distanceRewardFactor = 0.5

        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if not pygame.get_init():
            pygame.init()

        if self.headless:
            pygame.display.set_mode((1, 1))
            self.screen = None
        else:
            self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)

        if not self.headless:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None

        self.background = None
        self.background_norm = None
        self.background_flash = None

        self.fruit = None
        self.pause = Pause(not train_mode)
        self.level = 0
        self.lives = 0
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCaptured = []
        self.fruitNode = None
        self.mazedata = MazeData()

        self.last_node = None
        self.last_direction = STOP

        self.font = pygame.font.Font(None, 24)
        self.fitness_surface = None

        self.max_bfs_distance = 1

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level % 5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def startGame(self):
        self.mazedata.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.mazedata.obj.name + ".txt",
                                       self.mazedata.obj.name + "_rotation.txt")
        self.setBackground()

        # Nodi
        self.nodes = NodeGroup(self.mazedata.obj.name + ".txt")
        self.mazedata.obj.setPortalPairs(self.nodes)
        self.mazedata.obj.connectHomeNodes(self.nodes)

        # Pac-Man
        self.pacman = Pacman(self.nodes.getNodeFromTiles(*self.mazedata.obj.pacmanStart),
                             train_mode=self.train_mode)

        # Pellets
        self.pellets = PelletGroup(self.mazedata.obj.name + ".txt")

        # Fantasmi
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(0, 3)))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(4, 3)))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 3)))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(*self.mazedata.obj.addOffset(2, 0)))

        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.mazedata.obj.denyGhostsAccess(self.ghosts, self.nodes)

        self.compute_max_bfs_distance()

        if self.train_mode:
            self.pause.paused = False
            self.textgroup.hideText()

        self.last_node = self.pacman.node
        self.last_direction = self.pacman.direction
        self.prevDistanceToClosestPellet = None

    def compute_max_bfs_distance(self):
        max_dist = 0
        all_nodes = list(self.nodes.nodesLUT.values())
        for node in all_nodes:
            visited = set()
            queue = [(node, 0)]
            visited.add(node)
            while queue:
                curr_node, dist = queue.pop(0)
                if dist > max_dist:
                    max_dist = dist
                for neigh in curr_node.neighbors.values():
                    if neigh and neigh not in visited:
                        visited.add(neigh)
                        queue.append((neigh, dist + 1))
        if max_dist <= 0:
            max_dist = 1
        self.max_bfs_distance = max_dist

    def update(self):
        if not self.headless:
            dt = self.clock.tick(60) / 1000.0
        else:
            dt = self.fixed_dt

        self.textgroup.update(dt)
        self.pellets.update(dt)

        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit:
                self.fruit.update(dt)

            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        # Idle penalty
        if self.train_mode and self.pacman.alive and not self.pause.paused:
            self.timeSinceLastPellet += dt
            if self.timeSinceLastPellet > self.idlePelletThreshold:
                self.fitness -= self.idlePenalty
                self.pacman.die()
                self.game_over = True

        if self.pacman.alive and not self.pause.paused:
            self.pacman.update(dt)
        else:
            self.pacman.update(dt)  # animazione di morte

        current_node = self.pacman.node
        current_dir = self.pacman.direction

        # Ricontrollo input rete neurale a ogni cambio di nodo/direzione
        recalc_inputs = False
        if (current_node != self.last_node) or (current_dir != self.last_direction) or (current_dir == STOP):
            recalc_inputs = True

        # Reward se ci avviciniamo al pellet
        if self.train_mode and self.pacman.alive and not self.pause.paused:
            distPellet = self.get_bfs_distance_to_closest_pellet(current_node)
            if distPellet is not None:
                if self.prevDistanceToClosestPellet is not None:
                    diff = self.prevDistanceToClosestPellet - distPellet
                    if diff > 0:
                        self.fitness += diff * self.distanceRewardFactor
                self.prevDistanceToClosestPellet = distPellet

        # Attivazione rete neurale
        if self.train_mode and self.pacman.alive and not self.pause.paused and (self.net is not None):
            if recalc_inputs:
                input_data = self.get_relative_vision_input()
                if not self.headless:
                    print("AI INPUT:", input_data)
                output = self.net.activate(input_data)
                move_index = output.index(max(output))
                if move_index == 0:
                    desired_direction = self.pacman.direction
                elif move_index == 1:
                    desired_direction = OPPOSITE[self.pacman.direction]
                elif move_index == 2:
                    desired_direction = TURN_RIGHT[self.pacman.direction]
                else:
                    desired_direction = TURN_LEFT[self.pacman.direction]

                # Penalità piccola se l'IA va contro un muro (applicata ogni frame in cui è stuck davanti ad esso)
                if current_node and current_node.neighbors.get(desired_direction, None) is None:
                    self.fitness -= 0.15

                self.pacman.ai_direction = desired_direction

        self.last_node = current_node
        if current_dir != STOP:
            self.last_direction = current_dir

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod:
            afterPauseMethod()

        self.checkEvents()

        if self.train_mode:
            if not self.pacman.alive or self.lives < 0:
                self.game_over = True

        if not self.headless:
            self.render()

    # FUNZIONI DI BFS
    def get_bfs_distance_to_closest_pellet(self, start_node):
        if not start_node:
            return None
        visited = set([start_node])
        queue = [(start_node, 0)]
        while queue:
            node, dist = queue.pop(0)
            if self.has_pellet_at_node(node):
                return dist
            for neigh in node.neighbors.values():
                if neigh and neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, dist+1))
        return None

    def get_bfs_distance_between_nodes(self, start_node, end_node):
        if not start_node or not end_node:
            return None
        if start_node == end_node:
            return 0
        visited = set([start_node])
        queue = [(start_node, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node == end_node:
                return dist
            for neigh in node.neighbors.values():
                if neigh and neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, dist+1))
        return None

    def has_pellet_at_node(self, node):
        for pellet in self.pellets.pelletList:
            if pellet.position == node.position:
                return True
        return False


    def get_angle_to_nearest_pellet(self, pacman_node, pacman_direction):
        if not pacman_node:
            return 0.0
        px, py = pacman_node.position.x, pacman_node.position.y
        closest_pellet = None
        min_dist_sq = float('inf')

        for pellet in self.pellets.pelletList:
            dx = pellet.position.x - px
            dy = pellet.position.y - py
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pellet = pellet
        if not closest_pellet:
            return 0.0

        import math
        def angle_of_vector(vx, vy):
            return math.degrees(math.atan2(vy, vx))

        # Direzione di Pac-Man come vettore
        dir_vec = {
            UP:    (0, -1),
            DOWN:  (0, 1),
            LEFT:  (-1, 0),
            RIGHT: (1, 0),
            STOP:  (0, 0)
        }.get(pacman_direction, (0, 0))

        pac_dir_angle = angle_of_vector(dir_vec[0], dir_vec[1])
        dx = closest_pellet.position.x - px
        dy = closest_pellet.position.y - py
        pellet_angle = angle_of_vector(dx, dy)

        angle = pellet_angle - pac_dir_angle
        # angolo range [-180,180]
        if angle > 180:
            angle -= 360
        elif angle <= -180:
            angle += 360

        # angolo normalizzato in [-1, 1]
        return angle / 180.0

    def get_relative_vision_input(self):
        """
        Esempio: 11 input, TUTTI normalizzati in [-1,1].
        (front, left, right, back, ghostFreight, BFS pellet, angle pellet,
         BFS ghost1..ghost4)
        """
        # Funzione di supporto per mappare input [0,1] in [-1,1]
        def scale_01_to_m11(x):
            return 2.0 * x - 1.0

        current_node = self.pacman.node
        if not current_node:
            # Se Pac-Man non ha nodo di riferimento, ritorniamo 11 zeri, (Non dovrebbe mai avvenire)
            return [0.0] * 11

        if self.pacman.direction == STOP:
            forward_dir = self.last_direction
        else:
            forward_dir = self.pacman.direction

        back_dir = OPPOSITE[forward_dir]
        right_dir = TURN_RIGHT[forward_dir]
        left_dir = TURN_LEFT[forward_dir]

        dirs = [forward_dir, left_dir, right_dir, back_dir]

        def node_to_value(node):
            """
            Restituisce un valore in [0,1], poi sarà scalato a [-1,1].
            - 1.0 => muro (nessun neighbor)
            - 0.5 => fantasma
            - 0.0 => pellet o fruit o "vuoto" generico
            """
            if node is None:
                return 1.0  # "muro" / no access
            for ghost in self.ghosts:
                if ghost.node == node:
                    return 0.5  # fantasma
            for pellet in self.pellets.pelletList:
                if pellet.position == node.position:
                    return 0.0  # pellet
            if self.fruit and self.fruit.node == node:
                return 0.0  # fruit
            return 0.0     # altrimenti vuoto

        inputs_4dirs = []
        for d in dirs:
            neighbor = current_node.neighbors.get(d, None)
            val_01 = node_to_value(neighbor)
            val_m11 = scale_01_to_m11(val_01)
            inputs_4dirs.append(val_m11)

        # Fantasmi in stato FREIGHT?
        is_any_ghost_edible = any(ghost.mode.current == FREIGHT for ghost in self.ghosts)
        # 0 -> non edible, 1 -> edible => poi scala
        edible_01 = 1.0 if is_any_ghost_edible else 0.0
        edible_m11 = scale_01_to_m11(edible_01)

        # Distanza BFS pellet (0..1) -> (-1..1)
        dist_pellet = self.get_bfs_distance_to_closest_pellet(current_node)
        if dist_pellet is None:
            dist_pellet_norm = 1.0
        else:
            dist_pellet_norm = min(dist_pellet / self.max_bfs_distance, 1.0)
        dist_pellet_m11 = scale_01_to_m11(dist_pellet_norm)

        # Angolo pellet è già in [-1,1]
        angle = self.get_angle_to_nearest_pellet(current_node, self.pacman.direction)
        angle_norm = angle  # giusto per leggibilità

        # Distanze BFS verso i fantasmi
        ghost_distances_m11 = []
        for ghost in self.ghosts:
            gnode = ghost.node
            dist_g = self.get_bfs_distance_between_nodes(current_node, gnode)
            if dist_g is None:
                ghost_dist_norm = 1.0
            else:
                ghost_dist_norm = min(dist_g / self.max_bfs_distance, 1.0)
            ghost_distances_m11.append(scale_01_to_m11(ghost_dist_norm))

        # Assicuriamoci di avere 4 fantasmi al massimo
        while len(ghost_distances_m11) < 4:
            # Se non ci sono abbastanza fantasmi, simuliamo distanza massima
            ghost_distances_m11.append(scale_01_to_m11(1.0))

        return inputs_4dirs + [edible_m11, dist_pellet_m11, angle_norm] + ghost_distances_m11

    def checkEvents(self):
        if not self.train_mode:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        if self.pacman.alive:
                            self.pause.setPause(playerPaused=True)
                            if not self.pause.paused:
                                self.textgroup.hideText()
                                self.showEntities()
                            else:
                                self.textgroup.showText(PAUSETXT)
        else:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if pellet.name == POWERPELLET:
                self.fitness += 120.0
                self.ghosts.startFreight()
            else:
                self.fitness += 15.0

            self.timeSinceLastPellet = 0.0

            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)

            self.pellets.pelletList.remove(pellet)

            if self.pellets.isEmpty():
                # Fine livello
                self.fitness += 2000.0
                if self.train_mode:
                    self.game_over = True
                else:
                    pygame.quit()
                    sys.exit()

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.fitness += 150.0
                    self.textgroup.addText(str(ghost.points), WHITE,
                                           ghost.position.x, ghost.position.y,
                                           8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    # Pac-Man muore
                    self.fitness -= 100.0
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTXT)
                            if self.train_mode:
                                self.game_over = True
                            else:
                                self.pause.setPause(pauseTime=3,
                                                    func=self.restartGame)
                        else:
                            self.pause.setPause(pauseTime=3, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20),
                                   self.level)
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.fitness += 50.0
                self.textgroup.addText(str(self.fruit.points), WHITE,
                                       self.fruit.position.x, self.fruit.position.y,
                                       8, time=1)
                fruitCaptured = any(f.get_offset() == self.fruit.image.get_offset()
                                    for f in self.fruitCaptured)
                if not fruitCaptured:
                    self.fruitCaptured.append(self.fruit.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def restartGame(self):
        self.lives = 0
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)
        self.fitness = 0.0

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def render(self):
        if self.headless:
            return
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)

        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))

        for i in range(len(self.fruitCaptured)):
            x = SCREENWIDTH - self.fruitCaptured[i].get_width() * (i + 1)
            y = SCREENHEIGHT - self.fruitCaptured[i].get_height()
            self.screen.blit(self.fruitCaptured[i], (x, y))

        if self.train_mode:
            text = f"FITNESS: {int(self.fitness)}"
            self.fitness_surface = self.font.render(text, True, FITNESS_COLOR)
            text_rect = self.fitness_surface.get_rect(center=(FITNESS_POS[0], FITNESS_POS[1]))
            self.screen.blit(self.fitness_surface, text_rect)

        pygame.display.update()

#ESTENSIONE PER IL TRAINING "A STEP"
class GameControllerStep(GameController):
    """
    2 step:
      step=1 -> muri + pellet, nessun fantasma;
                se Pac-Man tenta di muoversi in un muro -> muore.
      step=2 -> mappa completa (muri, pellet, fantasmi).
    """
    def __init__(self, step=1, **kwargs):
        super().__init__(**kwargs)
        self.step = step

    def startGame(self):
        super().startGame()

        if self.step == 1:
            # Rimuoviamo i fantasmi
            self.ghosts.ghosts = []

    def update(self):
        # Se step=1, controlliamo se Pac-Man va verso un muro
        if self.step == 1 and self.pacman.alive and not self.pause.paused:
            desired_dir = self.pacman.ai_direction if self.pacman.ai_direction != STOP else self.pacman.direction
            if self.pacman.node:
                neighbor = self.pacman.node.neighbors.get(desired_dir, None)
                if neighbor is None:
                    # Muro -> Pac-Man muore
                    self.fitness -= 100
                    self.pacman.die()
                    self.game_over = True
                    return  # saltiamo il resto dell'update

            # In step=1, si premia il rimanere in vita evitando i muri
            self.fitness += 1.0

        super().update()


#FUNZIONI DI NEAT
def eval_genomes_visual(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameController(train_mode=True, net=net, config=config,
                              headless=False)
        game.startGame()
        while not game.game_over:
            game.update()
        genome.fitness = game.fitness

def eval_genomes_headless(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameController(train_mode=True, net=net, config=config,
                              headless=True, fixed_dt=1.0/60.0)
        game.startGame()
        while not game.game_over:
            game.update()
        genome.fitness = game.fitness

def evaluate_single_genome(genome, config):
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameController(train_mode=True, net=net, config=config,
                              headless=True, fixed_dt=1.0/60.0)
        game.startGame()
        while not game.game_over:
            game.update()
        fitness = game.fitness
    except:
        fitness = 0
    return fitness

def run_neat_visual(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes_visual, 150)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.draw_net(config, winner, True)

def run_neat_headless_sequential(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes_headless, 150)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.draw_net(config, winner, True)

def run_neat_headless_parallel(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_single_genome)
    winner = p.run(pe.evaluate, 150)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.draw_net(config, winner, True)


def replay_genome(config_file, genome_path="winner.pkl"):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    genomes = [(1, genome)]
    visualize.draw_net(config, genome, True)
    eval_genomes_visual(genomes, config)


#  FUNZIONI DI VALUTAZIONE GENOMI PER TRAINING A "STEP"
def evaluate_single_genome_step1(genome, config):
    fitness = 0.0  # Default come float
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameControllerStep(step=1, train_mode=True, net=net,
                                  config=config, headless=True,
                                  fixed_dt=1.0/60.0)
        game.startGame()
        while not game.game_over:
            game.update()
        fitness = float(game.fitness)  # Forza conversione a float
    except Exception as e:
        print(f"Error in STEP1 evaluation: {str(e)}")
        fitness = 0.0
    return fitness

def evaluate_single_genome_step2(genome, config):
    fitness = 0.0  # Default come float
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = GameControllerStep(step=2, train_mode=True, net=net,
                                  config=config, headless=True,
                                  fixed_dt=1.0/60.0)
        game.startGame()
        while not game.game_over:
            game.update()
        fitness = float(game.fitness)  # Forza conversione a float
    except Exception as e:
        print(f"Error in STEP2 evaluation: {str(e)}")
        fitness = 0.0
    return fitness


def run_stepwise_training(config_file, n_gen_step1=15, n_gen_step2=15):
    """
    - STEP 1: muri+pellet, no fantasmi, punizione se tocca muro.
    - STEP 2: scenario completo.
    Versione parallelizzata con fix per NoneType.
    """
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    #STEP 1
    print("\n*** STEP 1 ***")
    p1 = neat.Population(config)
    p1.add_reporter(neat.StdOutReporter(True))
    stats1 = neat.StatisticsReporter()
    p1.add_reporter(stats1)

    # Valutazione parallela
    pe1 = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_single_genome_step1)
    winner_step1 = p1.run(pe1.evaluate, n_gen_step1)

    with open("winner_step1.pkl", "wb") as f:
        pickle.dump(winner_step1, f)

    #STEP 2
    print("\n*** STEP 2 ***")
    p2 = neat.Population(config)

    # Clona il best genome dello step 1 per ogni membro della popolazione
    for gid in list(p2.population.keys()):
        new_genome = copy.deepcopy(winner_step1)
        new_genome.key = gid  # Imposta l'ID univoco
        new_genome.fitness = 0.0  # RESET del fitness
        p2.population[gid] = new_genome

    p2.species.speciate(config, p2.population, p2.generation)

    p2.add_reporter(neat.StdOutReporter(True))
    stats2 = neat.StatisticsReporter()
    p2.add_reporter(stats2)

    # Valutazione parallela con controllo del fitness
    pe2 = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_single_genome_step2)
    winner_step2 = p2.run(pe2.evaluate, n_gen_step2)

    with open("winner_step2.pkl", "wb") as f:
        pickle.dump(winner_step2, f)

    print("\n*** TRAINING 2 STEP COMPLETATO ***")
    print(" - Step1 salvato in winner_step1.pkl")
    print(" - Step2 salvato in winner_step2.pkl")

    # Plot statistiche e rete
    visualize.plot_stats(stats2, ylog=False, view=True)
    visualize.draw_net(config, winner_step2, True)
    pygame.quit()


#MAIN
if __name__ == "__main__":
    multiprocessing.freeze_support()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")

    print("***********************************")
    print("          Pac-Man AI v1.0          ")
    print("***********************************\n")

    print("1) Allenare in modalità classica NEAT (con scelte).")
    print("2) Giocare manualmente.")
    print("3) Far giocare l'AI salvata (winner.pkl).")
    print("4) Eseguire il training a step (2 step).")
    choice = input("Scelta: ")

    if choice == "1":
        if not os.path.exists(config_path):
            print("File di config NEAT non trovato (neat-config.txt).")
            sys.exit(1)

        print("\nScegli la modalità:")
        print("a) Visuale (lenta).")
        print("b) Headless sequenziale.")
        print("c) Headless parallelizzata.")
        subc = input("Scelta: ")
        if subc == "a":
            run_neat_visual(config_path)
        elif subc == "b":
            run_neat_headless_sequential(config_path)
        elif subc == "c":
            run_neat_headless_parallel(config_path)
        else:
            print("Scelta non valida.")
            sys.exit(1)

    elif choice == "2":
        if not pygame.get_init():
            pygame.init()
        game = GameController()
        game.startGame()
        while True:
            game.update()

    elif choice == "3":
        try:
            replay_genome(config_path)
        except:
            print('Non trovo "winner.pkl". Rinominare o generare il file prima di usarlo.')
            input('Invio per uscire...')
            sys.exit()

    elif choice == "4":
        if not os.path.exists(config_path):
            print("File di config NEAT non trovato (neat-config.txt).")
            sys.exit(1)

        run_stepwise_training(config_path,
                              n_gen_step1=10,
                              n_gen_step2=80)
    else:
        print("Scelta non valida.")
        sys.exit()
