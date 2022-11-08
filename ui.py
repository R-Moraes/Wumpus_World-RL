import pygame
import sys
from os import path
sys.path.append(path.join(path.abspath('.'), 'gym_game'))
sys.path.append(path.join(path.abspath('.'), 'gym_game','env'))
from custom_env import CustomEnv

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)

class Gui_environment:
    def __init__(self, size_env):
        self.window_height = 650
        self.window_width = 650
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.width_cave = self.window_width - 100
        self.height_cave = self.window_height - 100
        self.pix_init = 50
        self.size_env = size_env
        self.block_size = int(self.width_cave/self.size_env)
        self.img_wumpus = None
        self.img_gold = None
        self.img_pit = None
        self.img_agent = None
        self.img_breeze = None

        self.load_piece()
    
    def draw_board(self):
        pygame.init()

        for x in range(self.pix_init, self.width_cave, self.block_size):
            for y in range(self.pix_init, self.height_cave, self.block_size):
                rect = pygame.Rect(x,y,self.block_size, self.block_size)
                pygame.draw.rect(self.screen, BLACK, rect, 1)

    def calc_pos_img(self, x, y, size_env):
        limit = ((size_env-1)*self.block_size) + self.pix_init
        coord_x = self.pix_init + (y * self.block_size)
        coord_y = self.pix_init + (((size_env-1)-x) * self.block_size)

        return min(coord_x, limit), coord_y

    def load_piece(self):
        abs_dir = path.abspath(path.join('gym_game', 'gui', 'img'))
        size_split = 5
        self.img_wumpus = [pygame.image.load(path.join(abs_dir, 'wumpus.png'))]
        self.img_wumpus = [pygame.transform.scale(img, (self.block_size-size_split, self.block_size-size_split))
                            for img in self.img_wumpus]

        self.img_gold = [pygame.image.load(path.join(abs_dir, 'gold.png'))]
        self.img_gold = [pygame.transform.scale(img, (self.block_size-size_split, self.block_size-size_split))
                            for img in self.img_gold]

        self.img_pit = [pygame.image.load(path.join(abs_dir, 'pit.png'))]
        self.img_pit = [pygame.transform.scale(img, (self.block_size-size_split, self.block_size-size_split))
                            for img in self.img_pit]

        self.img_agent = [pygame.image.load(path.join(abs_dir, 'agent.png')), 
                        pygame.image.load(path.join(abs_dir, 'agent_arrow.png'))]
        self.img_agent = [pygame.transform.scale(img, (self.block_size-size_split, self.block_size-size_split))
                            for img in self.img_agent]
        
        self.img_breeze = [pygame.image.load(path.join(abs_dir, 'breeze.png'))]
        self.img_breeze = [pygame.transform.scale(img, (self.block_size-size_split, self.block_size-size_split))
                            for img in self.img_breeze]

    def add_wumpus(self, coord):
        x, y = coord
        self.wumpus_coord = [self.calc_pos_img(x,y, self.size_env)]
        print('W', self.calc_pos_img(x, y, self.size_env))

    def add_pits(self, coord):
        self.pits_coord = [self.calc_pos_img(x,y, self.size_env) for x,y in coord]
        print('P', [self.calc_pos_img(x,y, self.size_env) for x,y in coord])

    def add_gold(self, coord):
        x, y = coord
        self.gold_coord = [self.calc_pos_img(x,y, self.size_env)]
        print('G',self.calc_pos_img(x, y, self.size_env))

    def add_breeze(self, coord):
        x, y = coord
        self.breeze_coord = [self.calc_pos_img(x,y, self.size_env)]
        print('Bre', self.calc_pos_img(x, y, self.size_env))

    def add_agent(self, coord:tuple):
        x,y = coord
        self.agent_coord = [self.calc_pos_img(x, y, self.size_env)]
        print('A', self.calc_pos_img(x, y, self.size_env))
    
    def update_board(self):
        self.screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                sys.exit()

        self.draw_board()

        for coordinate in self.wumpus_coord:
            self.screen.blit(self.img_wumpus[0], coordinate)

        for coordinate in self.gold_coord:
            self.screen.blit(self.img_gold[0], coordinate)

        for coordinate in self.pits_coord:
            self.screen.blit(self.img_pit[0], coordinate)

        for coordinate in self.agent_coord:
            self.screen.blit(self.img_agent[1], coordinate)

        # for coordinate in self.breeze_coord:
        #     self.screen.blit(self.img_breeze[0], coordinate)
        
        pygame.display.update()

if __name__ == '__main__':
    dict_max_steps = {4: 100, 8: 150, 10: 200} #size environment is key and value is amount max steps
    dict_values_seed = {4: 123, 8: 99, 10: 917} #size environment is key and value is values seed
    dim = 10
    
    env = CustomEnv(nrow=dim,ncol=dim, max_steps=dict_max_steps[dim], value_seed=dict_values_seed[dim])
    gui = Gui_environment(size_env=dim)

    gui.add_agent(env.environment.board.components['Agent'].pos)
    gui.add_wumpus(env.environment.board.components['Wumpus'].pos)
    coord_pits = [env.environment.board.components[f'Pit{i}'].pos for i in range(gui.size_env-1)]
    gui.add_pits(coord_pits)
    gui.add_gold(env.environment.board.components['Gold'].pos)
    # gui.add_agent(env.environment.board.components['Agent'].pos)

    env.render()
    print('Coord Gold:', env.environment.board.components['Gold'].pos)
    print('Coord Agent:', env.environment.board.components['Agent'].pos)
    print('Coord Wumpus:', env.environment.board.components['Wumpus'].pos)
    print('Coord Pits:', coord_pits)

    while True:
        gui.update_board()


