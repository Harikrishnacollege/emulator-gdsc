import pygame
import sys
import pickle
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


SCREEN_WIDTH = 64
SCREEN_HEIGHT = 32
MEMORY_SIZE = 4096
REGISTER_COUNT = 16
STACK_SIZE = 16
KEY_COUNT = 16

SCALE = 10  # Final OpenGL viewport size: 640x320

fontset = [
    0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
    0x20, 0x60, 0x20, 0x20, 0x70,  # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
    0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
    0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
    0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
    0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
    0xF0, 0x80, 0xF0, 0x80, 0x80   # F
]


class Chip8:
    def __init__(self):
        # Memory, registers and other state variables.
        self.memory = [0] * MEMORY_SIZE
        self.v = [0] * REGISTER_COUNT       # V0-VF, with VF as flag.
        self.i = 0                          # Index register.
        self.pc = 0x200                     # Program counter starts at 0x200.
        self.gfx = [0] * (SCREEN_WIDTH * SCREEN_HEIGHT)
        self.delay_timer = 0
        self.sound_timer = 0
        self.stack = [0] * STACK_SIZE
        self.sp = 0                         # Stack pointer.
        self.key = [0] * KEY_COUNT          # CHIP-8 keypad state.

        # Load fontset into memory (starting at address 0).
        for i in range(len(fontset)):
            self.memory[i] = fontset[i]

    def load_program(self, program_bytes):
        """Load a CHIP-8 program (list/bytes of values) into memory starting at 0x200."""
        for i in range(len(program_bytes)):
            self.memory[0x200 + i] = program_bytes[i]


    def emulate_cycle(self):
        # Fetch the next opcode. CHIP-8 opcodes are 2 bytes.
        opcode = self.memory[self.pc] << 8 | self.memory[self.pc + 1]
        self.decode_and_execute(opcode)
        
        if self.delay_timer > 0:
            self.delay_timer -= 1
        if self.sound_timer > 0:
            self.sound_timer -= 1

    def decode_and_execute(self, opcode):
        # OPCODE DECODING with extended instruction support.
        if opcode == 0x00E0:
            # 00E0: Clear the display.
            self.gfx = [0] * (SCREEN_WIDTH * SCREEN_HEIGHT)
            self.pc += 2

        elif opcode == 0x00EE:
            # 00EE: Return from subroutine.
            self.sp -= 1
            self.pc = self.stack[self.sp]
            self.pc += 2

        elif opcode & 0xF000 == 0x1000:
            # 1NNN: Jump to address NNN.
            self.pc = opcode & 0x0FFF

        elif opcode & 0xF000 == 0x2000:
            # 2NNN: Call subroutine at NNN.
            self.stack[self.sp] = self.pc
            self.sp += 1
            self.pc = opcode & 0x0FFF

        elif opcode & 0xF000 == 0x3000:
            # 3XNN: Skip next instruction if Vx == NN.
            x = (opcode & 0x0F00) >> 8
            nn = opcode & 0x00FF
            self.pc += 4 if self.v[x] == nn else 2

        elif opcode & 0xF000 == 0x6000:
            # 6XNN: Set Vx = NN.
            x = (opcode & 0x0F00) >> 8
            self.v[x] = opcode & 0x00FF
            self.pc += 2

        elif opcode & 0xF000 == 0x7000:
            # 7XNN: Add NN to Vx (no carry).
            x = (opcode & 0x0F00) >> 8
            self.v[x] = (self.v[x] + (opcode & 0x00FF)) & 0xFF
            self.pc += 2

        elif opcode & 0xF000 == 0xA000:
            # ANNN: Set I = NNN.
            self.i = opcode & 0x0FFF
            self.pc += 2

        elif opcode & 0xF000 == 0xD000:
            # DXYN: Draw sprite at coordinate (Vx, Vy) with width 8 and height N.
            x = self.v[(opcode & 0x0F00) >> 8]
            y = self.v[(opcode & 0x00F0) >> 4]
            height = opcode & 0x000F
            self.v[0xF] = 0  # Reset collision flag.
            for row in range(height):
                pixel = self.memory[self.i + row]
                for col in range(8):
                    if (pixel & (0x80 >> col)) != 0:
                        idx = ((y + row) % SCREEN_HEIGHT) * SCREEN_WIDTH + ((x + col) % SCREEN_WIDTH)
                        if self.gfx[idx] == 1:
                            self.v[0xF] = 1
                        self.gfx[idx] ^= 1
            self.pc += 2

        else:
            print(f"Unknown opcode: {opcode:04X}")
            self.pc += 2

    def handle_keys(self):
        key_map = {
            pygame.K_1: 0x1, pygame.K_2: 0x2, pygame.K_3: 0x3, pygame.K_4: 0xC,
            pygame.K_q: 0x4, pygame.K_w: 0x5, pygame.K_e: 0x6, pygame.K_r: 0xD,
            pygame.K_a: 0x7, pygame.K_s: 0x8, pygame.K_d: 0x9, pygame.K_f: 0xE,
            pygame.K_z: 0xA, pygame.K_x: 0x0, pygame.K_c: 0xB, pygame.K_v: 0xF,
        }
        keys = pygame.key.get_pressed()
        for py_key, chip8_key in key_map.items():
            self.key[chip8_key] = 1 if keys[py_key] else 0

    def save_state(self, filename):
        state = {
            "memory": self.memory,
            "v": self.v,
            "i": self.i,
            "pc": self.pc,
            "gfx": self.gfx,
            "delay_timer": self.delay_timer,
            "sound_timer": self.sound_timer,
            "stack": self.stack,
            "sp": self.sp,
            "key": self.key,
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        print("State saved to", filename)

    def load_state(self, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        self.memory = state["memory"]
        self.v = state["v"]
        self.i = state["i"]
        self.pc = state["pc"]
        self.gfx = state["gfx"]
        self.delay_timer = state["delay_timer"]
        self.sound_timer = state["sound_timer"]
        self.stack = state["stack"]
        self.sp = state["sp"]
        self.key = state["key"]
        print("State loaded from", filename)


def init_gl_texture():
    """Create and set up an OpenGL texture to render the CHIP-8 display."""
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # Create an empty texture of size SCREEN_WIDTH x SCREEN_HEIGHT.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    return texture

def update_gl_texture(texture, gfx):
    """
    Update the OpenGL texture with the current display from the CHIP-8's gfx buffer.
    The gfx buffer is a list of 0s and 1s that represent pixels.
    """
    # Create a NumPy array with shape (SCREEN_HEIGHT, SCREEN_WIDTH, 3):
    # white pixels (255,255,255) where gfx == 1 and black (0,0,0) otherwise.
    pixel_data = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    # Reshape the linear gfx array into (SCREEN_HEIGHT, SCREEN_WIDTH)
    gfx_array = np.array(gfx, dtype=np.uint8).reshape((SCREEN_HEIGHT, SCREEN_WIDTH))
    pixel_data[gfx_array == 1] = [255, 255, 255]
    glBindTexture(GL_TEXTURE_2D, texture)
    # Update the texture with the pixel data.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)

def draw_texture_quad(texture):
    """
    Draw a full-screen quad with the given texture.
    Coordinates are set up so the texture maps onto the entire OpenGL viewport.
    """
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)
    # Bottom-left vertex (with texture coordinate 0,0)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(-1.0, -1.0)
    # Bottom-right vertex (with texture coordinate 1,0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, -1.0)
    # Top-right vertex (with texture coordinate 1,1)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)
    # Top-left vertex (with texture coordinate 0,1)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(-1.0, 1.0)
    glEnd()
    glDisable(GL_TEXTURE_2D)

def main():
    pygame.init()
    # Create a window with an OpenGL context.
    screen = pygame.display.set_mode(
        (SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE),
        pygame.OPENGL | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("CHIP-8 Emulator (OpenGL Renderer)")

    glViewport(0, 0, SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClearColor(0, 0, 0, 1)  

    clock = pygame.time.Clock()

    chip8 = Chip8()
    sample_program = [
        0x60, 0x00,  # 6000: Set V0 = 0
        0x61, 0x00,  # 6100: Set V1 = 0
        0xA2, 0x0A,  # A20A: Set I = 0x20A (assume sprite data is here)
        0xD0, 0x11,  # D011: Draw sprite at (V0,V1) with height 1
        0x12, 0x00   # 1200: Jump to address 0x200 (infinite loop)
    ]
    chip8.load_program(sample_program)

    # Create the OpenGL texture for the display.
    texture = init_gl_texture()

    running = True
    while running:
        # Event handling.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    chip8.save_state("chip8_state.pkl")
                elif event.key == pygame.K_F9:
                    chip8.load_state("chip8_state.pkl")

        chip8.handle_keys()
        chip8.emulate_cycle()
        update_gl_texture(texture, chip8.gfx)
        draw_texture_quad(texture)

        pygame.display.flip()
        clock.tick(60)  

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
 