import sys
import math
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # loop over all variables
        for var in self.domains:
            # initialize set to store words that are consistent with variable length
            consistent_var = set()

            # loop over all words in domain and add them to set if they match length
            for word in self.domains[var]:
                if len(word) == var.length:
                    consistent_var.add(word)

            # update variable domain so that it contains consistent words only
            self.domains[var] = consistent_var

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # initialize variable to track if any change was made
        revised = False
        overlaps = self.crossword.overlaps[x, y]

        if overlaps:
            i, j = overlaps
            # set to store values in x domain that are inconsistent
            inconsistent_x = set()
            for x_word in self.domains[x]:
                # keep track of whether x value can find a corresponding y value
                match = False
                for y_word in self.domains[y]:
                    try:
                        if x_word[i] == y_word[j]:
                            match = True
                            break
                    except:
                        continue
                if not match:
                    inconsistent_x.add(x_word)
                    revised = True
            # remove inconsistent x values from x domain
            self.domains[x] -= inconsistent_x
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # initialize queue list. if no arcs argument add all arcs i.e. tuples of variables and their neighbors
        queue = []
        if arcs is None:
            for var in self.crossword.variables:
                neighbors = self.crossword.neighbors(var)
                if neighbors:
                    for neighbor in neighbors:
                        queue.append((var, neighbor))
        else:
            queue = arcs

        # continuous loop while queue is not empty
        while queue:
            # remove first element of queue and unpack
            x, y = queue.pop(0)
            if self.revise(x, y):
                # if revised domain of x is empty, then unsolvable
                if len(self.domains[x]) == 0:
                    return False
                x_neighbors = self.crossword.neighbors(x)
                x_neighbors.remove(y)
                # since x domain was changed, check arc consistency with all x neighbors except y
                for neighbor in x_neighbors:
                    queue.append((neighbor, x))
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # loop over all variables
        for var in self.crossword.variables:

            # check if variable is not assigned a value. if not, assignment incomplete
            if var not in assignment or not assignment[var]:
                return False

        # if all variables pass check, assignment is complete
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # check that each assignment has a distinct value
        if len(assignment) != len(set(assignment.values())):
            return False

        for var in assignment:

            # check for equality between assignment value length and variable length
            if len(assignment[var]) != var.length:
                return False

            # check for conflicts between neighbors
            neighbors = self.crossword.neighbors(var)
            for neighbor in neighbors:
                i, j = self.crossword.overlaps[var, neighbor]
                if neighbor in assignment:
                    if assignment[var][i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # initialize list to store words and number of values eliminated
        eliminations = []
        neighbors = self.crossword.neighbors(var)
        for word in self.domains[var]:

            # counter variable to keep track of values eliminated for each word in variable domain
            eliminated = 0

            # loop over all neighbors
            for neighbor in neighbors:

                # we are only concerned with unassigned neighbors
                if neighbor not in assignment:

                    # loop over every word in neighbor's domain and check for conflicts
                    for neighbor_word in self.domains[neighbor]:
                        i, j = self.crossword.overlaps[var, neighbor]
                        if word[i] != neighbor_word[j]:
                            eliminated += 1
            # store (word, # of values eliminated) in list
            eliminations.append((word, eliminated))

        # sort list by ascending number of values eliminated
        eliminations.sort(key=lambda x: x[1])
        # return list of words only
        return [word[0] for word in eliminations]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # two variables keep track of which unassigned crossword variable has the fewest values in its domain
        min = math.inf
        min_var = None
        for var in self.crossword.variables:
            if var not in assignment:
                # if lower value found, assign min and min_var accordingly
                if len(self.domains[var]) < min:
                    min = len(self.domains[var])
                    min_var = var
                # if value is equal to already existing min, assign min_var to variable with more neighbors
                elif len(self.domains[var]) == min:
                    if len(self.crossword.neighbors(var)) > len(self.crossword.neighbors(min_var)):
                        min_var = var
        return min_var

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # return assignment if complete
        if self.assignment_complete(assignment):
            return assignment

        # select unassigned variable
        var = self.select_unassigned_variable(assignment)

        # loop through ordered list of domain values i.e. possible words
        for value in self.order_domain_values(var, assignment):
            # try assigning words to variable
            assignment[var] = value
            if self.consistent(assignment):
                # recursively call backtrack to test further unassigned variables after original
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                # remove variable from assignment if no solution found
                assignment.pop(var, None)

        # return None if variable fails to satisfy constraints
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
