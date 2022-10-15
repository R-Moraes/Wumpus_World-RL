import numpy as np
# VALUE_SEED = 99 # 123 -- 4x4, 99 -- 8x8,  917 -- 10x10

class BoardPiece:
    def __init__(self, name, code, pos, direction='E') -> None:
        self.name = name        #name of the piece
        self.code = code        #An ASCII character to display on the board
        self.pos = pos          #2-tuple e.g. (1,4)

class PieceAgent(BoardPiece):
    def __init__(self, name, code, pos, max_steps, direction='E') -> None:
        super().__init__(name, code, pos)
        self.direction = direction    #N-North, S-South, E-East, W-West
        self.has_gold = False
        self.wumpus_alive = True
        self.last_action = None
        self.amount_arrows = 1
        self.has_arrow = True
        self.max_steps = max_steps
        self.amount_steps = max_steps
    
    def turn_left(self):
        if self.direction == 'N':
            self.direction = 'W'
        elif self.direction == 'S':
            self.direction = 'E'
        elif self.direction == 'E':
            self.direction = 'N'
        elif self.direction == 'W':
            self.direction = 'S'
    
    def turn_right(self):
        if self.direction == 'N':
            self.direction = 'E'
        elif self.direction == 'S':
            self.direction = 'W'
        elif self.direction == 'E':
            self.direction = 'S'
        elif self.direction == 'W':
            self.direction = 'N'
    
    def grab(self, pos_gold):
        if self.pos == pos_gold:
            self.has_gold = True
    
    def shoot(self, pos_wumpus, size_env):
        next_coord = self.coord_direction_looking(size_env)
        if next_coord == pos_wumpus:
            self.wumpus_alive = False

    #check coordinate that according to the direction I am looking
    def coord_direction_looking(self, size_env):
        direction = self.direction
        row, col = self.pos
        if direction == 'N':
            row = min(row+1, size_env-1)
        elif direction == 'S':
            row = max(row-1, 0)
        elif direction == 'E':
            col = min(col+1, size_env-1)
        elif direction == 'W':
            col = max(col-1, 0)

        return row, col

class WumpusBoard:
    def __init__(self, size_env, max_steps, value_seed):
        self.size_env = size_env
        self.components = {}
        self.sensations = {'breeze':set(), 'stench':set(), 'glitter':set()}
        self.max_steps = max_steps
        self.value_seed = value_seed
        self.array_seed = self.generate_array_seed()
    
    def generate_array_seed(self):
        np.random.seed(self.value_seed)
        value = (self.size_env-1) + 2
        array_seed = np.random.randint(low=1, high=100, size=value)

        return array_seed
    
    def rand_Pair(self):
        row = np.random.randint(0, self.size_env)
        col = np.random.randint(0, self.size_env)

        return row, col
    
    def add_Piece(self, name, code, pos, direction=None):
        if code == 'A':
            new_piece = PieceAgent(name, code, pos, self.max_steps, direction)
        else:
            new_piece = BoardPiece(name, code, pos)
        self.components[name] = new_piece
    
    def move_Piece(self, name, pos):
        self.components[name] = pos
    
    def get_board_str(self, matrix):
        matrix_str = ''
        for i in range(self.size_env-1,-1,-1):
            matrix_str += '|'
            for j in range(self.size_env):
                matrix_str += f' {matrix[i,j]} |'
            matrix_str +='\n'
        
        return matrix_str
    
    def get_board_str_two(self, matrix):
        matrix_list = []
        matrix_str = ''
        for i in range(self.size_env):
            matrix_str += '|'
            for j in range(self.size_env):
                matrix_str += f' {matrix[i,j]} |'
            # matrix_str +='\n'
            matrix_list.append(matrix_str)
            matrix_str = ''
        return matrix_list
    
    def coord_is_valid(self, pos:tuple):
        x,y = pos
        if (x < self.size_env and x >= 0) and (y < self.size_env and y >= 0):
                return True
        else:
            return False

    def add_sensations(self, coord:tuple):
        coord_sensations = set()
        x,y = coord
        if self.coord_is_valid((x+1,y)):
            coord_sensations.add((x+1,y))
        if self.coord_is_valid((x-1,y)):
            coord_sensations.add((x-1,y))
        if self.coord_is_valid((x,y+1)):
            coord_sensations.add((x,y+1))
        if self.coord_is_valid((x,y-1)):
            coord_sensations.add((x,y-1))
        
        return coord_sensations

    def get_matrix_sensations(self):
        matrix_sen = np.zeros((self.size_env, self.size_env), dtype='<U2')
        matrix_sen[:] = 'E'
        for name, piece in self.components.items():
            if piece.code == 'P':
                list_s = self.add_sensations(piece.pos)
                #brise --- br
                self.sensations['breeze'] = self.sensations['breeze'] | list_s
            elif piece.code == 'W':
                list_s = self.add_sensations(piece.pos)
                #stench --- st
                self.sensations['stench'] = self.sensations['stench'] | list_s
            elif piece.code == 'G':
                list_s = set([piece.pos])
                #glitter --- gl
                self.sensations['glitter'] = self.sensations['glitter'] | list_s
        
        for name, list_pos in self.sensations.items():
            for x,y in list(list_pos):
                matrix_sen[x,y] = name
        
        return matrix_sen
    
    def get_matrix_env(self):
        matrix_env = np.zeros((self.size_env, self.size_env), dtype='<U2')
        matrix_env[:] = 'E'
        agent_pos = self.components['Agent'].pos
        
        for name, piece in self.components.items():
            if agent_pos == piece.pos:
                '''AGENTE ESTA NA CASA EM QUE HA OURO, WUMPUS OU POÇO'''
                matrix_env[piece.pos] = 'A'
            else:
                if piece.code == 'W' and not self.components['Agent'].wumpus_alive:
                    piece.code = 'X'
                matrix_env[piece.pos] = piece.code
        
        return matrix_env
    
    def get_sensations(self, pos:tuple):
        dict_sensations = { cood: [] for value in self.sensations.values() for cood in value}
        for key, value in self.sensations.items():
            if pos in dict_sensations.keys():
                if pos in value:
                    dict_sensations[pos].append(key)
            else:
                dict_sensations[pos] = ['E']
        
        return dict_sensations[pos]


class WumpusWorld:
    def __init__(self, size, max_steps, value_seed) -> None:
        self.board = WumpusBoard(size, max_steps, value_seed)
        self.board.add_Piece('Agent', 'A', (0,0), 'E')
        self.generation_environment()
    
    def generation_environment(self):
        self.init_grid_rand()
        while not self.environment_is_valid():
            self.init_grid_rand()
        _ = self.board.get_matrix_sensations()
    
    def position_is_valid(self, pos):
        positions = []
        for name, piece in self.board.components.items():
            positions.append(piece.pos)
        
        if pos in positions:
            return False
        else:
            return True
    
    def init_grid_rand(self):
        idx = 0
        pieces = [ (f'Pit{i}', 'P') for i in range(0, self.board.size_env-1)] + [('Gold', 'G'), ('Wumpus', 'W')]
        for name, cod in pieces:
            np.random.seed(self.board.array_seed[idx])
            pos = self.board.rand_Pair()
            while not self.position_is_valid(pos):
                pos = self.board.rand_Pair()
            self.board.add_Piece(name, cod, pos)
            idx += 1
    
    def environment_is_valid(self):
        return self.depthSearch((0,0))
    
    def getGraph(self, )->dict:
        grafo = {}
        matrix = self.board.get_matrix_env()
        n = self.board.size_env

        for i in range(n):
            for j in range(n):
                cima, baixo, direita, esquerda = (i+1,j), (i-1,j), (i,j+1), (i,j-1)
                nodes = []
                if i == 0:
                    if matrix[cima[0]][cima[1]] != 'P': nodes.append(cima)
                    if j == 0:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                    elif j == n-1:
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)
                    else:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)

                elif i == n-1:
                    if matrix[baixo[0]][baixo[1]] != 'P': nodes.append(baixo)
                    if j == 0:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                    elif j == n-1:
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)
                    else:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)
                        
                else:
                    if matrix[baixo[0]][baixo[1]] != 'P': nodes.append(baixo)
                    if matrix[cima[0]][cima[1]] != 'P': nodes.append(cima)
                    if j == 0:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                    elif j == n-1:
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)
                    else:
                        if matrix[direita[0]][direita[1]] != 'P': nodes.append(direita)
                        if matrix[esquerda[0]][esquerda[1]] != 'P': nodes.append(esquerda)
                grafo.update({(i,j):nodes})
                
        return grafo
              
    def depthSearch(self, start:object):
        matrix = self.board.get_matrix_env()
        graph = self.getGraph()
        visiteds = [start]
        not_visiteds = [start]
        while not_visiteds:
            current = not_visiteds.pop()
            for neighbor in graph[current]:
                if neighbor not in visiteds:
                    visiteds.append(neighbor)
                    not_visiteds.append(neighbor)
                    x,y=neighbor
                    if matrix[x][y] == 'G': return True
        return False
    
    def reset_environment(self):
        self.board.components['Agent'].pos = (0,0)
        self.board.components['Agent'].direction = 'E'
        self.board.components['Agent'].has_gold = False
        self.board.components['Agent'].has_arrow = True
        self.board.components['Agent'].wumpus_alive = True
        self.board.components['Agent'].last_action = None
        self.board.components['Agent'].amount_arrows = 1
        self.board.components['Agent'].amount_steps = self.board.components['Agent'].max_steps
        self.board.components['Gold'].code = 'G'
        self.board.components['Wumpus'].code = 'W'
        return self.observe()
    
    def verify_direction(self):
        direction = self.board.components['Agent'].direction
        row, col = self.board.components['Agent'].pos
        if direction == 'N':
            row = min(row+1, self.board.size_env-1)
        elif direction == 'S':
            row = max(row-1, 0)
        elif direction == 'E':
            col = min(col+1, self.board.size_env-1)
        elif direction == 'W':
            col = max(col-1, 0)

        self.board.components['Agent'].pos = (row, col)

    def invalid_move(self):
        direction = self.board.components['Agent'].direction
        row, col = self.board.components['Agent'].pos
        # if direction == 'N':
        #     row += 1
        # elif direction == 'S':
        #     row -= 1
        # elif direction == 'E':
        #     col += 1
        # elif direction == 'W':
        #     col -= 1

        return (row, col)
    
    def distance_euclidean(self):
        agent = self.board.components['Agent']
        gold = self.board.components['Gold']
        return round(np.sqrt((agent.pos[0]-gold.pos[0])**2 + (agent.pos[1]-gold.pos[1])**2), 5)

    def move(self, action):
        self.board.components['Agent'].amount_steps -= 1
        self.board.components['Agent'].last_action = action
        #FORWARD, TURN_LEFT, TURN_RIGHT, GRAB, SHOOT
        if action == 'FORWARD':
            self.verify_direction()
        elif action == 'TURN_LEFT':
            self.board.components['Agent'].turn_left()
        elif action == 'TURN_RIGHT':
            self.board.components['Agent'].turn_right()
        elif action == 'GRAB':
            pos_gold = self.board.components['Gold'].pos
            self.board.components['Agent'].grab(pos_gold)
        elif action == 'SHOOT':
            if self.board.components['Agent'].has_arrow:
                pos_wumpus = self.board.components['Wumpus'].pos
                self.board.components['Agent'].shoot(pos_wumpus, self.board.size_env)
            # self.board.components['Agent'].amount_arrows -= 1
            # self.board.components['Agent'].has_arrow = False
    
    def convert_sensations_in_matrix(self):
        #0 - Breeze 1-Stench 3-Glitter
        state_sensations = [0,0,0]
        current_row, current_col = self.board.components['Agent'].pos
        matrix = self.board.get_sensations((current_row, current_col))
        for sen in matrix:
            if sen == 'breeze':
                state_sensations[0] = 1
            elif sen == 'stench':
                state_sensations[1] = 1
            elif sen == 'glitter' and not self.board.components['Agent'].has_gold:
                state_sensations[2] = 1
        
        return state_sensations
    
    def observe(self):
        current_row, current_col = self.get_pos_agent()
        amount_arrow = self.board.components['Agent'].amount_arrows if self.board.components['Agent'].amount_arrows == 1 else 0
        wumpus_alive = 1 if self.board.components['Agent'].wumpus_alive else 0
        dict_directions = {'N':0, 'S':1, 'E':2, 'W':3}
        direction = self.board.components['Agent'].direction
        state_env = [current_row * self.board.size_env + current_col, dict_directions[direction]]
        state_sensations = self.convert_sensations_in_matrix()
        distance_gold = self.distance_euclidean()
        information_for_agent = [amount_arrow, wumpus_alive]
        state = state_env + state_sensations + information_for_agent
        state = np.array(state)

        return current_row * self.board.size_env + current_col
    
    def evaluate(self):
        current_pos_agent = self.get_pos_agent()
        last_action = self.board.components['Agent'].last_action
        piece_coord = [piece.pos for piece in self.board.components.values() if piece.code == 'P']
        
        if current_pos_agent == (0,0) and self.board.components['Agent'].has_gold:
            return 1000
        elif self.board.components['Wumpus'].pos == current_pos_agent and self.board.components['Agent'].wumpus_alive:
            return -100
        elif current_pos_agent in piece_coord:
            return -100
        elif last_action == 'GRAB':
            if self.board.components['Gold'].pos == current_pos_agent and self.board.components['Gold'].code == 'G':
                    self.board.components['Gold'].code = 'E'
                    return 500
            else: return -1
        elif last_action == 'SHOOT':
            if self.board.components['Agent'].has_arrow:
                self.board.components['Agent'].amount_arrows -= 1
                self.board.components['Agent'].has_arrow = False
                if self.shoot_wumpus()==self.board.components['Wumpus'].pos and self.board.components['Wumpus'].code == 'W':
                    self.board.components['Wumpus'].code == 'E'
                    return 100
                else:
                    return -1
            else:
                return -1
        elif last_action == 'FORWARD':
            x,y = self.invalid_move()
            if not self.board.coord_is_valid((x,y)):
                return -2

        return -1
    
    def is_done(self):
        current_pos_agent = self.get_pos_agent()
        last_action = self.board.components['Agent'].last_action
        piece_coord = [piece.pos for piece in self.board.components.values() if piece.code == 'P']
        if current_pos_agent==(0,0) and self.board.components['Agent'].has_gold:
            return True
        elif self.board.components['Wumpus'].pos == current_pos_agent and self.board.components['Agent'].wumpus_alive:
            return True
        elif current_pos_agent in piece_coord:
            return True
        elif self.board.components['Agent'].amount_steps == 0:
            return True
        
        return False

    def get_pos_agent(self):
        return self.board.components['Agent'].pos

    def shoot_wumpus(self):
        row,col = self.get_pos_agent()

        direction = self.board.components['Agent'].direction
        if direction == 'N':
            row = min(row+1, self.board.size_env-1)
        elif direction == 'S':
            row = max(row-1, 0)
        elif direction == 'E':
            col = min(col+1, self.board.size_env-1)
        elif direction == 'W':
            col = max(col-1, 0)

        return (row, col)     


# if __name__ == '__main__':
#     env = WumpusWorld(4, 100)
#     env.board.components['Pit0'].pos = (3,1)
#     env.board.components['Pit1'].pos = (1,2)
#     env.board.components['Pit2'].pos = (0,2)
#     env.board.components['Gold'].pos = (2,1)
#     env.board.components['Wumpus'].pos = (0,3)
#     env.board.components['Agent'].pos = (1,3)
    
#     mat = env.board.get_matrix_env()
#     mat_sen = env.board.get_matrix_sensations()
#     print(env.board.get_board_str(mat))
#     print(env.evaluate())
#     print(env.observe())
#     print(env.distance_euclidean())
#     env.move('TURN_RIGHT')
#     env.move('SHOOT')
#     print(env.evaluate())
#     print(env.observe())
#     mat = env.board.get_matrix_env()
#     print(env.board.get_board_str(mat))
#     env.move('FORWARD')
#     print(env.board.components['Agent'].pos)
#     print(env.evaluate())
#     print(env.observe())
#     mat = env.board.get_matrix_env()
#     print(env.board.get_board_str(mat))
#     env.board.components['Agent'].pos = (1,3)
#     env.move('SHOOT')
#     print(env.evaluate())
#     print(env.observe())
#     mat = env.board.get_matrix_env()
#     print(env.board.get_board_str(mat))
#     for i in range(env.board.size_env):
#         for j in range(env.board.size_env):
#             env.board.components['Agent'].pos = (i,j)
#             current_row, current_col = env.board.components['Agent'].pos
#             state = current_row * env.board.size_env + current_col
#             print(f'pos: ({current_row},{current_col}) and state: {state}')
    



