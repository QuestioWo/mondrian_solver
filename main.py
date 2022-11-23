import copy
import random
import time

from typing import (Tuple, List)
from enum import Enum

import cv2
import numpy as np
from numba import jit

MONDRIAN_COLORS = [
	(1,240,255),
	(1,1,255),
	(253,1,1),
	(249,249,249),
	(58,48,48)
]

class ActionType(Enum):
	SPLIT = 0
	MERGE = 1


class ActionDirection(Enum):
	VERTICAL = 2
	HORIZONTAL = 3


class Rectangle :
	def __init__(self, x: int, y: int, width: int, height: int) :
		self.x = x
		self.y = y
		self.width = width
		self.height = height

	def __eq__(self, other) :
		return (
			self.x == other.x
			and self.y == other.y
			and self.width == other.width
			and self.height == other.height)

	def getSides(self) -> List[Tuple[Tuple[int, int]]] :
		return [
			((self.x, self.y), (self.x + self.width, self.y)),
			((self.x, self.y), (self.x, self.y + self.height)),
			((self.x + self.width, self.y), (self.x, self.y + self.height)),
			((self.x, self.y + self.height), (self.x + self.width, self.y))
		]

	def __repr__(self) -> str:
		return "Rectangle(%d,%d,%d,%d)" % (self.x, self.y, self.width, self.height)

	def __hash__(self) -> int :
		# Create hash for efficiently describing a Rectangle of specific dimensions
		return (self.x + self.y * 31 + self.width * 41 + self.height * 53)


class Action:
	def __init__(self, incidentRectangles: List[int], t: ActionType, index: int, direction: ActionDirection) :
		self.incidentRectangles = incidentRectangles
		self.t = t
		self.index = index
		self.direction = direction


def computeStateHash(rectangles: List[Rectangle]) -> int :
	# Create hash for state to allow for efficient compariosn with othe states
	return sum([(i * 10**3) * hash(r) for (i, r) in enumerate(rectangles)])


def flipRectangles(rectangles: List[Rectangle], direction: ActionDirection, a: int) -> List[Rectangle] :
	new_rects = None
	if direction == ActionDirection.VERTICAL :
		new_rects = [Rectangle(a - (r.x + r.width), r.y, r.width, r.height) for r in rectangles]
	else : # direction == ActionDirection.HORIZONTAL
		new_rects = [Rectangle(r.x, a - (r.y + r.height), r.width, r.height) for r in rectangles]

	return new_rects


def rotateRectangles90Clockwise(rectangles: List[Rectangle], a: int) -> List[Rectangle] :
	new_rects = []
	# flip rectangles horzontally
	new_rects = flipRectangles(rectangles, ActionDirection.HORIZONTAL, a)

	# transpose all rectangles
	for r in new_rects :
		r.x, r.y = r.y, r.x
		r.width, r.height = r.height, r.width

	return new_rects


class State:
	def __init__(self, rectangles: List[Rectangle], depth: int, a: int) :
		self.rectangles = sorted(rectangles, key=lambda r: r.x * 10000 + r.y)
		self.depth = depth
		self.a = a

		horizontally_flipped_rects = flipRectangles(self.rectangles, ActionDirection.HORIZONTAL, a)
		rotated_180 = flipRectangles(horizontally_flipped_rects, ActionDirection.VERTICAL, a)

		self.hash = min([ # all congruent/rotational and mirrored variants of the same state
			computeStateHash(self.rectangles),
			computeStateHash(horizontally_flipped_rects),
			computeStateHash(flipRectangles(self.rectangles, ActionDirection.VERTICAL, a)),
			computeStateHash(rotated_180), # 180
			computeStateHash(rotateRectangles90Clockwise(self.rectangles, a)), # 270
			computeStateHash(rotateRectangles90Clockwise(rotated_180, a)), # 90
		])

	def isValid(self) -> bool :
		for i, r1 in enumerate(self.rectangles) :
			for j, r2 in enumerate(self.rectangles) :
				if j <= i :
					continue
				
				if isCongruent(r1, r2) :
					return False

		return True

	def __eq__(self, other) -> bool :
		return hash(self) == hash(other)

	def __hash__(self) -> int :
		return self.hash

	def generatePossibleActions(self) -> List[Action] :
		actions = []
		for i, rect in enumerate(self.rectangles) :
			actions += [Action([i], ActionType.SPLIT, j, ActionDirection.VERTICAL) for j in range(1, rect.width)]
			actions += [Action([i], ActionType.SPLIT, j, ActionDirection.HORIZONTAL) for j in range(1, rect.height)]

			currentSides = rect.getSides()

			merge_actions = {}

			for j, rect2 in enumerate(self.rectangles) :
				if i == j :
					continue

				otherSides = rect2.getSides()

				equality_sides = [(s in otherSides) for s in currentSides]
				if any(equality_sides) :
					# index and direction do not matter
					merge_actions[sorted([i, j])] = Action([i, j], ActionType.MERGE, 0, 0)
					
			actions += list(merge_actions.values())

		return actions


def splitRectangle(rect1: Rectangle, index: int, direction: ActionDirection) -> List[Rectangle] :
	if direction == ActionDirection.HORIZONTAL :
		old_height = rect1.height
		rect1.height = index
		return [rect1, Rectangle(rect1.x, rect1.y + rect1.height, rect1.width, old_height - rect1.height)]
	
	else : # direction == VERTICAL
		old_width = rect1.width
		rect1.width = index
		return [rect1, Rectangle(rect1.x + rect1.width, rect1.y, old_width - rect1.width, rect1.height)]


def mergeRectangle(rect1: Rectangle, rect2: Rectangle) -> List[Rectangle] :
	rect1_sides = rect1.getSides()
	rect2_sides = rect2.getSides()
	
	new_x = new_y = new_width = new_height = None 

	for i, s1 in enumerate(rect1_sides) :
		for s2 in rect2_sides :
			if s1 != s2 :
				continue

			match i :
				case 0 :
					new_x = rect2.x # or rect1.x
					new_y = rect2.y
					new_width = rect2.width # or rect1.width
					new_height = rect1.height + rect2.height

				case 1 :
					new_x = rect2.x
					new_y = rect2.y # or rect1.y
					new_width = rect1.width + rect2.width
					new_height = rect2.height # or rect1.height

				case 2 :
					new_x = rect1.x
					new_y = rect1.y # or rect2.y
					new_width = rect1.width + rect2.width
					new_height = rect1.height # or rect1.height
				
				case 3 :
					new_x = rect1.x # or rect2.x
					new_y = rect1.y
					new_width = rect1.width # or rect2.width
					new_height = rect1.height + rect2.height

			break

	return [Rectangle(new_x, new_y, new_width, new_height)]


def deepCopyRectangles(rectangles: List[Rectangle]) -> List[Rectangle] :
	return [Rectangle(r.x, r.y, r.width, r.height) for r in rectangles]


def actOnState(initial_state: State, action: Action) -> State :
	rectangles = deepCopyRectangles(initial_state.rectangles)

	resulting_rectangles: List[Rectangle] = []

	if (action.t == ActionType.SPLIT) :
		resulting_rectangles = splitRectangle(rectangles.pop(action.incidentRectangles[0]), action.index, action.direction)
	else : # action.t == ActionType.MERGE
		# use max and min for pop as otherwise indices in action.incidentRectangles won't be correct as list size will change
		resulting_rectangles = mergeRectangle(rectangles.pop(max(action.incidentRectangles)), rectangles.pop(min(action.incidentRectangles)))

	rectangles += resulting_rectangles

	return State(rectangles, initial_state.depth + 1, initial_state.a)


def calculateMondrian(s: State) :
	if not s.isValid() :
		return float("inf")
	
	largest = max(s.rectangles, key = lambda r: r.width * r.height)
	largestSize = largest.width * largest.height

	smallest = min(s.rectangles, key = lambda r: r.width * r.height)
	smallestSize = smallest.width * smallest.height

	return largestSize - smallestSize


def isCongruent(rect1: Rectangle, rect2: Rectangle) -> bool :
	return ((rect1.width == rect2.width and rect1.height == rect2.height)
		or (rect1.width == rect2.height and rect1.height == rect2.width))


def showState(a: int, rects: List[Rectangle], name: str = "best") :
	img = np.zeros((a * 100, a * 100, 3), dtype=np.uint8)

	copy_modrian_colours = []

	for rect in rects :
		if len(copy_modrian_colours) == 0 :
			copy_modrian_colours = copy.copy(MONDRIAN_COLORS)

		bgr_index = random.choice(range(len(copy_modrian_colours)))
		bgr = copy_modrian_colours.pop(bgr_index)
		
		img = cv2.rectangle(img, (rect.x * 100, rect.y * 100), (rect.x * 100 + rect.width * 100, rect.y * 100 + rect.height * 100), bgr, -1)
		img = cv2.rectangle(img, (rect.x * 100, rect.y * 100), (rect.x * 100 + rect.width * 100, rect.y * 100 + rect.height * 100), (0,0,0), 3)

	cv2.imwrite(name + ".png", img)


# @jit(parallel=True, forceobj=True)
def iterateBFS(M: int, explored_states: dict, stateQueue: List[State]) -> None :
	while not len(stateQueue) == 0 :
		current_state = stateQueue.pop(0)

		if current_state.depth >= M :
			break

		new_states = []
		for possible_action in current_state.generatePossibleActions() :
			acted_on_state = actOnState(current_state, possible_action)

			if not hash(acted_on_state) in explored_states :
				unexplored = True
				explored_states[hash(acted_on_state)] = acted_on_state
				new_states.append(acted_on_state)

		stateQueue += new_states


def SolveMondrian(a: int, M: int, show: bool = True) -> None :
	print("a := %d; M := %d" % (a, M))

	initial_state = State([Rectangle(0, 0, a, a)], 0, a)
	stateQueue = [initial_state]
	explored_states = {}
	
	iterateBFS(M, explored_states, stateQueue)

	ranked_explored_states = sorted(list(explored_states.values()), key=lambda s : calculateMondrian(s))

	score = calculateMondrian(ranked_explored_states[0])

	print("Best mondrian score := %d\n" % score)

	if show :
		showState(a, ranked_explored_states[0].rectangles, "best_%dX%d=%d" % (a, M,score))

	return score

		
def main() :
	a_s = [8, 12, 16, 20]
	M_s = [2, 3, 4, 5, 6, 7, 8]

	time_table = [[('', ''), *zip(a_s, a_s)]]

	for M in M_s :
		holding_arr = [(M, M)]
		for a in a_s :
			start = time.time()
			score = SolveMondrian(a, M)
			end = time.time()

			holding_arr.append((end - start, score)) 

		time_table.append(holding_arr)
		
	print("\nTimes:")
	for h in time_table :
		for t in h :
			print(t[0], end='\t')
		print("")

	print("\nScores:")
	for h in time_table :
		for t in h :
			print(t[1], end='\t')
		print("")


if __name__ == "__main__" :
	main()