import copy
import random
import time
import multiprocessing as mp

from typing import (Tuple, List)
from enum import Enum

import cv2
import numpy as np
from sortedcontainers import SortedList

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
	x: int
	y: int
	width: int
	height: int

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
	incidentRectangles : List[Rectangle]
	t: ActionType
	index: int
	direction: ActionDirection

	def __init__(self, incidentRectangles: List[int], t: ActionType, index: int, direction: ActionDirection) :
		self.incidentRectangles = incidentRectangles
		self.t = t
		self.index = index
		self.direction = direction


def computeStateHash(rectangles: List[Rectangle]) -> int :
	# Create hash for state to allow for efficient comparison with othe states
	return sum([(10**(3 * i)) * hash(r) for (i, r) in enumerate(rectangles)])


def flipRectangles(rectangles: List[Rectangle], direction: ActionDirection, a: int) -> List[Rectangle] :
	new_rects = None
	if direction == ActionDirection.VERTICAL :
		new_rects = [Rectangle(a - (r.x + r.width), r.y, r.width, r.height) for r in rectangles]
	else : # direction == ActionDirection.HORIZONTAL
		new_rects = [Rectangle(r.x, a - (r.y + r.height), r.width, r.height) for r in rectangles]

	return new_rects


def transposeRectangles(rectangles: List[Rectangle]) -> List[Rectangle] :
	return [Rectangle(r.y, r.x, r.height, r.width) for r in rectangles]


class State:
	rectangles : List[Rectangle]
	depth: int
	a: int
	hash: int

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
			computeStateHash(transposeRectangles(horizontally_flipped_rects)), # 270
			computeStateHash(transposeRectangles(flipRectangles(rotated_180, ActionDirection.HORIZONTAL, a))), # 90
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


def expandDepth(current_state: State) -> List[State] :
	return [actOnState(current_state, possible_action) for possible_action in current_state.generatePossibleActions()]
	

def updateQueueWithUnique(new_queue: List[List[State]], explored_states: SortedList, scores_states: dict) -> List[State] :
	unique_queue = []
	for i in range(len(new_queue)) :
		for j in range(len(new_queue[i])) :
			curr_state = new_queue[i][j]
			
			not_explored = (explored_states.count(hash(curr_state)) == 0)
			
			if not_explored :
				explored_states.add(hash(curr_state))
				scores_states[calculateMondrian(curr_state)] = curr_state
				unique_queue.append(curr_state)

	return unique_queue


def iterateBFS(M: int, scores_states: dict, stateQueue: List[State]) -> None :
	explored_states = SortedList()
	for _ in range(M) :
		new_depth_queue_map = []
		with mp.Pool() as pool :
			new_depth_queue_map = pool.map(expandDepth, stateQueue)

		stateQueue = updateQueueWithUnique(new_depth_queue_map, explored_states, scores_states)

	return explored_states


def iterateIDDFS(M : int, scores_states: dict, stateList : List[State]) -> None :
	explored_states = SortedList()
	while len(stateList) > 0 :
		# Instead, treat stateList as a stack, meaning that the deepest depth will be explored before a parallel branch is
		current_state = stateList.pop()

		child_states = [actOnState(current_state, possible_action) for possible_action in current_state.generatePossibleActions()]

		for i in range(len(child_states)) :
			child_state = child_states[i]
			
			not_explored = (explored_states.count(hash(child_state)) == 0)

			if not_explored :
				explored_states.add(hash(child_state))
				scores_states[calculateMondrian(child_state)] = child_state
				if child_state.depth != M:
					stateList.append(child_state)


def SolveMondrian(a: int, M: int, show: bool = True) -> None :
	print("a := %d; M := %d" % (a, M))

	initial_state = State([Rectangle(0, 0, a, a)], 0, a)
	stateQueue = [initial_state]
	scores_states = {}
	
	# Disabling BFS as although it is faster, it is more memory hungry
	# to the extent of using all available RAM
	
	# iterateBFS(M, scores_states, stateQueue)
	iterateIDDFS(M, scores_states, stateQueue)

	ranked_scores = sorted(list(scores_states.keys()))

	score = ranked_scores[0]

	print("Best mondrian score := %d\n" % score)

	if show :
		best_state = scores_states[score]
		showState(a, best_state.rectangles, "best_%dX%d=%d" % (a, M, score))

	return score

		
def main() :
	a_s = [8, 12, 16, 20]
	M_s = range(2, 20)

	# a_s = [10]
	# M_s = [2, 3, 4]
	
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