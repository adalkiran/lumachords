import numpy as np

from lumachords.hands_detector import HandMidiNumRange


class StaffMapping:
    MIDI_NUM_C4 = 60
    MIDI_NUM_E4 = 65
    MIDI_NUM_G3 = 55
    STAFF_RIGHT_HAND = 0
    STAFF_LEFT_HAND = 1

    @staticmethod
    def determine_staff_idx(
        hands_midi_num_ranges: list[HandMidiNumRange], 
        default_split_midi_num: int, 
        midi_num: int
    ) -> int:
        hands_count = len(hands_midi_num_ranges) if hands_midi_num_ranges is not None else 0
        if hands_count < 1 or hands_count > 2:
            return __class__.STAFF_LEFT_HAND if midi_num < (default_split_midi_num or __class__.MIDI_NUM_C4) else __class__.STAFF_RIGHT_HAND
        if hands_count == 1:
            single_hand = hands_midi_num_ranges[0]
            if midi_num < single_hand.start_midi_num:
                return __class__.STAFF_LEFT_HAND
            else:
                return __class__.STAFF_RIGHT_HAND
        elif hands_count == 2:
            left_hand, right_hand = hands_midi_num_ranges
            if left_hand is None or (right_hand is not None and left_hand.end_midi_num < right_hand.start_midi_num - 2):
                if left_hand is None:
                    return __class__.STAFF_LEFT_HAND if right_hand is not None and midi_num < right_hand.start_midi_num else __class__.STAFF_RIGHT_HAND
                if midi_num <= left_hand.end_midi_num:
                    return __class__.STAFF_LEFT_HAND
                if right_hand is not None and midi_num >= right_hand.start_midi_num:
                    return __class__.STAFF_RIGHT_HAND

                left_mid = (left_hand.start_midi_num + left_hand.end_midi_num) / 2
                right_mid = (right_hand.start_midi_num + right_hand.end_midi_num) / 2
                return __class__.STAFF_LEFT_HAND if abs(midi_num - left_mid) <= abs(midi_num - right_mid) else __class__.STAFF_RIGHT_HAND
            elif midi_num > left_hand.end_midi_num:
                return __class__.STAFF_RIGHT_HAND
            return __class__.STAFF_LEFT_HAND
    
    @staticmethod
    def determine_clef_key(midi_nums: np.ndarray, prev_clef_key: str) -> str:
        midi_mean = midi_nums.mean()
        if prev_clef_key is not None:
            if midi_mean <= __class__.MIDI_NUM_C4:
                return "F" if midi_mean < __class__.MIDI_NUM_C4 else prev_clef_key
            else:
                if any(midi_nums > __class__.MIDI_NUM_E4):
                    if any(midi_nums < __class__.MIDI_NUM_G3):
                        return "F"
                    else:
                        return "G"
                else:
                    return "F" if any(midi_nums < __class__.MIDI_NUM_C4) else "G"
        else:
            return "F" if midi_mean < __class__.MIDI_NUM_C4 else "G"
        return "G"
