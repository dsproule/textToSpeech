from CONSTS import *
import numpy as np

class Caster:
	'''
		Caster()
		converts strings to id maps and vice versa
	'''

	# str_to_map()
	# Converts a string to character mapping
	# returns: a list of integers in format [<s>, string, <e>]
	def str_to_map(self, string, showInvalid=False):
		string = string.lower()
		mapping = [START_ID]
		for char in string:
			charID = ord(char) - 96			# Will form alphabet into a:1 - z:26, space:32
			if charID >= 1 and charID <= 26:
				mapping.append(charID)
			elif charID == -64:
				mapping.append(SPACE_ID)
			elif showInvalid:
				print(f"WARNING: Invalid character ({char}:{charID}) processed. Was not encoded but did appear")
		mapping.append(END_ID)

		return np.array(mapping)
	
	# map_to_str()
	# Converts a character mapping to a string
	def map_to_str(self, idMap, showInvalid=False):
		mapping = []
		for charID in idMap:
			if charID is START_ID:
				mapping.append('<s>')
			elif charID is END_ID:
				mapping.append('<e>')
			elif charID == SPACE_ID:
				mapping.append(' ')
			elif charID >= 1 and charID <= 26:
				mapping.append(chr(charID + 96))
			elif showInvalid:
				print(f"WARNING: Invalid character ({chr(charID)}:{charID}) processed. Was not encoded but did appear")
		return np.array(mapping)

	# ctc_strip()
	# Cleans a character mapping according to the ctc loss function
	# rules so a loss value can be calculated
	def ctc_strip(self, idMap):
		mapping, lastChar, i = [], '', 0
		for curChar in idMap:
			if curChar != lastChar:
				lastChar = curChar
				if curChar != BLANK_ID:
					mapping.append(curChar)

		return np.array(mapping)

	# padded_map()
	# Converts to a string mapping and fills the rest with blank chars
	# returns: a list of integers in format [<s>, string, <e>, 0, ..., 0], label length
	def padded_str_to_map(self, string, padLen):
		mapping = np.zeros(padLen)
		mapping[:len(string) + 2] = self.str_to_map(string)

		return mapping, len(string) + 2