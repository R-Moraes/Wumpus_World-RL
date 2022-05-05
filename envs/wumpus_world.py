import numpy as np

class BoardPiece:
    def __init__(self, name, code, pos, direction='E') -> None:
        self.name = name        #name of the piece
        self.code = code        #An ASCII character to display on the board
        self.pos = pos          #2-tuple e.g. (1,4)

class PieceAgent(BoardPiece):
    def __init__(self, name, code, pos, direction='E') -> None:
        super().__init__(name, code, pos)
        self.direction = direction    #N-North, S-South, E-East, W-West
        self.has_gold = False
        self.wumpus_alive = True
        self.last_action = None
        self.amount_arrows = 1
    
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
    
    def shoot(self, pos_wumpus):
        if self.pos == pos_wumpus:
            self.wumpus_alive = False

class WumpusBoard:
    def __init__(self, size_env):
        self.size_env = size_env
        self.components = {}
        self.sensations = {'breeze':set(), 'stench':set(), 'glitter':set()}
    
    def rand_Pair(self):
        row = np.random.randint(0, self.size_env)
        col = np.random.randint(0, self.size_env)

        return row, col
    
    def add_Piece(self, name, code, pos, direction=None):
        if code == 'A':
            new_piece = PieceAgent(name, code, pos, direction)
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

        for name, piece in self.components.items():
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
    def __init__(self, size) -> None:
        self.board = WumpusBoard(size)
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
        pieces = [ (f'Pit{i}', 'P') for i in range(0, self.board.size_env-1)] + [('Gold', 'G'), ('Wumpus', 'W')]
        for name, cod in pieces:
            pos = self.board.rand_Pair()
            while not self.position_is_valid(pos):
                pos = self.board.rand_Pair()
            self.board.add_Piece(name, cod, pos)
    
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
        piece = self.board.components['Agent']
        piece.pos = (0,0)
        piece.direction = 'E'
        piece.has_gold = False
        piece.wumpus_alive = True
        piece.last_action = None
        piece.amount_arrows = 1
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

    def move(self, action):
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
            pos_wumpus = self.board.components['Wumpus'].pos
            self.board.components['Agent'].shoot(pos_wumpus)
            self.board.components['Agent'].amount_arrows -= 1
    
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
            elif sen == 'glitter':
                state_sensations[2] = 1
        
        return state_sensations
    
    def observe(self):
        # print(self.board.components['Agent'].pos)
        current_row, current_col = self.board.components['Agent'].pos
        amount_arrow = self.board.components['Agent'].amount_arrows
        wumpus_alive = self.board.components['Agent'].wumpus_alive
        state_env = [current_row * self.board.size_env + current_col]
        state_sensations = self.convert_sensations_in_matrix()
        information_for_agent = [amount_arrow, wumpus_alive]
        state = state_env + state_sensations + information_for_agent
        state = np.array(state)

        return state
    
    def evaluate(self):
        current_pos_agent = self.get_pos_agent()
        last_action = self.board.components['Agent'].last_action
        for piece in self.board.components.values():
            if piece.code == 'W':
                if piece.pos == current_pos_agent:
                    return -10
            elif piece.code == 'P':
                if piece.pos == current_pos_agent:
                    return -10
            elif last_action == 'GRAB':
                if piece.pos == current_pos_agent and piece.code == 'G':
                    return 10
            elif last_action == 'SHOOT':
                if self.board.components['Agent'].amount_arrows == 0:
                    return -1
                else:
                    if self.shoot_wumpus()==piece.pos and piece.code == 'W':
                        self.board.components['Agent'].wumpus_alive = False
                        self.board.components['Agent'].amount_arrows -= 1
                        return 10
                    else:
                        self.board.components['Agent'].amount_arrows -= 1
                        return -1
        return -1
    
    def is_done(self):
        current_pos_agent = self.get_pos_agent()
        for piece in self.board.components.values():
            if piece.code == 'G':
                if piece.pos == current_pos_agent:
                    return True
            elif piece.code == 'W':
                if piece.pos == current_pos_agent:
                    return True
            elif piece.code == 'P':
                if piece.pos == current_pos_agent:
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


if __name__ == '__main__':
    env = WumpusWorld(4)
    mat = env.board.get_matrix_env()
    mat_sen = env.board.get_matrix_sensations()
    print(env.board.get_board_str(mat))
    print(env.observe())
    env.reset_environment()



