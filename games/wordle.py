import numpy 
import string
from utils.utils import read_dataset
# from ..utils.utils import read_dataset

COLORS = {"Green" : "\033[32m", # Or \033[92m for bright green
          "Yellow": "\033[33m", # Or \033[93m for bright yellow
          "Gray"  : "\033[37m", 
          "Reset" : "\033[0m", 
          "White" : "\033[97m", 
          "Red"   : "\033[91m", 
          "DarkGray": "\033[90m", 
         }
BACKGROUND_COLORS = {"Green"  : "\033[42m", 
                     "Yellow" : "\033[43m", 
                     "Gray"   : "\033[47m", 
                     "DarkGray": "\033[100m", 
                     "Reset" : "\033[0m", 
                    }

class Wordle: 
    def __init__(self, mode="player", render_mode="simple", dataset_path="./dataset/wordle.csv", seed = None): 
        self.guess_remaining=6
        self.dataset_path = dataset_path
        self.seed = seed
        if mode not in ["player", "ai"]: 
            raise ValueError(f"Only support two modes: player, ai. Found {mode}")
        else: 
            self.mode = mode
        if render_mode not in ["simple", "rich", "train"]: 
            raise ValueError(f"Only support three render modes: simple, rich, train. Found {render_mode}")
        else: 
            self.render_mode = render_mode
        self.given_word = self.get_given_word()
        alphabet = list(string.ascii_lowercase)
        self.guessed_letter = {}
        for char in alphabet: 
            self.guessed_letter[char] = "Reset"
        self.guessed_words = ['', '', '', '', '', '']
        self.colors_hist = []

    def get_given_word(self): 
        self.words = read_dataset(self.dataset_path)[0]
        rng = numpy.random.default_rng(self.seed)
        word = rng.choice(self.words)
        return word

    def assign_color(self):
        colors = ['', '', '', '', '']
        letter_dict = {}
        for letter in self.given_word: 
            if letter not in letter_dict: 
                letter_dict[letter] = 1
            else: 
                letter_dict[letter] += 1

        for idx in range(5): 
            # Green 
            if self.player_guess[idx] == self.given_word[idx]: 
                letter_dict[self.player_guess[idx]] -= 1
                colors[idx] = "Green"
                self.guessed_letter[self.player_guess[idx].lower()] = "Green"
                continue 
        for idx in range(5): 
            # Gray
            if colors[idx]=="Green": 
                continue
            if self.player_guess[idx] not in self.given_word: 
                if self.guessed_letter[self.player_guess[idx].lower()] == "Reset":
                    self.guessed_letter[self.player_guess[idx].lower()] = "DarkGray" 
                colors[idx] = "Gray" 
                continue
            elif self.player_guess[idx] in self.given_word: 
                if letter_dict[self.player_guess[idx]] > 0: 
                    # Yellow 
                    if self.guessed_letter[self.player_guess[idx].lower()] != "Green": 
                        self.guessed_letter[self.player_guess[idx].lower()] = "Yellow"
                    letter_dict[self.player_guess[idx]] -= 1
                    colors[idx] = "Yellow"
                else: 
                    # Gray 
                    colors[idx] = "Gray"
                continue
        return colors

    def is_valid_guess(self): 
        if len(self.player_guess) != 5: 
            print("Only 5-letter word")
            return False
        if self.player_guess not in self.words: 
            print(f"Word {self.player_guess} not in dictionary")
            return False
        return True

    def is_win(self): 
        return self.player_guess == self.given_word

    def play(self): 
        if self.mode == "player": 
            # For debug 
            # print(self.given_word)
            while self.guess_remaining > 0: 
                self.player_guess = input("Enter your letter: ")
                while not self.is_valid_guess(): 
                    self.player_guess = input("Enter your letter: ")
                colors = self.assign_color()
                self.guessed_words[6-self.guess_remaining] = self.player_guess
                self.colors_hist.append(colors)
                self.render(colors)
                if self.is_win(): 
                    print("Win")
                    break
                self.guess_remaining -= 1
            if self.guess_remaining == 0: 
                print(f"Lose, the word is : {self.given_word}")

    def draw_row(self, letters:str|list|None=None, colors:list|None = None):
        if letters: 
            if len(letters) != 5: 
                raise ValueError("Word must have lenght of 5")
            if not colors: 
                raise ValueError("Please provide colors")
        top_bot_row = []
        mid_row = []
        for idx in range(5): 
            if letters is not None and colors is not None: 
                top_bot_row.append(COLORS[colors[idx]]+"+---+"+COLORS["Reset"])
                mid_row.append(COLORS[colors[idx]]+f"| {letters[idx]} |"+COLORS["Reset"])
            else:
                top_bot_row.append("+---+")
                mid_row.append("|   |")
        print(''.join(top_bot_row))
        print(''.join(mid_row))
        print(''.join(top_bot_row))

    def draw_title(self): 
        title = ['+-----------------------+', 
                 '|         WORDLE        |',
                 '+-----------------------+']
        for i in title: 
            print(i)

    def draw_alphabet(self): 
        out_text = ""
        # for char, color in self.guessed_letter.items(): 
        #     out_text += COLORS[color]+char+COLORS["Reset"]+" "
        # print(out_text)
        qwerty_grid = [
                      ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                      ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                      ['z', 'x', 'c', 'v', 'b', 'n', 'm']
                      ]
        for row in range(len(qwerty_grid)): 
            for char in qwerty_grid[row]: 
                out_text += " "+BACKGROUND_COLORS[self.guessed_letter[char]]+" "+char+" "+COLORS["Reset"]+" "
            if row == 0: 
                print(out_text)
                out_text = "  "
            if row == 1:
                out_text += " "
                print(out_text)
                out_text = "   "
            if row == 2: 
                out_text += "  "
                print(out_text)

    def clear_terminal(self): 
        print("\033[H\033[2J\033[3J", end="", flush=True)

    def render(self, colors): 
        if self.render_mode == "simple": 
            output_text = []
            for idx in range(5): 
                output_text.append(COLORS[colors[idx]]+self.player_guess[idx]+COLORS["Reset"])
            print(f"{self.guess_remaining} | {output_text[0]} | {output_text[1]} | {output_text[2]} | {output_text[3]} | {output_text[4]} |")
        elif self.render_mode == "rich": 
            self.clear_terminal()
            self.draw_title()
            for i in range(6):
                if self.guessed_words[i] == '': 
                    self.draw_row()
                else: 
                    self.draw_row(letters=self.guessed_words[i], colors=self.colors_hist[i])
            # print(self.guessed_words)
            # print(self.colors_hist)
            self.draw_alphabet()
        else: 
            pass

if __name__ == "__main__": 
    wordle = Wordle(render_mode="rich")
    wordle.play()
