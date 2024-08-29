from recurrent import Brain
from numpy.typing import NDArray
from codecs import decode


ASCII_RANGE = (0x0020, 0x007E)


def get_unicode_characters_by_ranges(
    ranges: list[tuple[int]] = [ASCII_RANGE],
) -> str:
    string_characters = list()
    for start, finish in ranges:
        if finish < start:
            raise ValueError('Start value of range is bigger thn finish value')
        for number in range(start, finish + 1):
            string_characters.append(chr(number))
    return ''.join(string_characters)


def translate_string_to_inputs_sequence(
    string: str, translating_string: str,
) -> list[list[float]]:
    inputs_sequence = list()
    for character in string:
        char_index = translating_string.find(character)
        inputs_sequence.append([char_index / (len(translating_string) - 1)])
    return inputs_sequence


def translate_outputs_sequence_to_string(
    outputs_sequence: list[list[float]], translating_string: str,
) -> str:
    characters = list()
    for output in outputs_sequence:
        index = round((len(translating_string) - 1) * output[-1])
        char = translating_string[index]
        characters.append(char)
    return ''.join(characters)

class ChatBot(Brain):
    def get_answer(self, question: str) -> str:
        ascii_question = repr(question)[1 : -1]
        input_sequence = translate_string_to_inputs_sequence(
            ascii_question, get_unicode_characters_by_ranges(),
        )
        output_sequence = self(input_sequence)
        ascii_answer = translate_outputs_sequence_to_string(
            output_sequence, get_unicode_characters_by_ranges(),
        )
        answer = decode(ascii_answer, 'unicode-escape')
        return answer

    def talk(self):
        pass


# Testing
if __name__ == '__main__':
    chatbot = ChatBot()
    answer = chatbot.get_answer('Хорошо в деревне летом!')
    print(answer)
