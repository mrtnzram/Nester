# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:05:07 2021

@author: Ramartin
Last edited: 1/29/2024 by kate snyder - JSON output includes wav file name modified from gzip file name
1/22/2024 - add export_to_json for UMAP workflow

# How to Run:
# Open a terminal
# use cd to navigate to where the python script is stored (or open the terminal in the directory that has the script)
# in the terminal type "python3 Scriptname.py" and it should just run!
# To change the targeted directory, at the bottom of the script change the 'directory' variable to be the folder containing the gzips

"""



import json
import numpy as np
import gzip
import pandas as pd
import os
import time
import multiprocessing as mp
from skimage.measure import label, regionprops
#from skimage.transform import rescale, resize, downscale_local_mean
import pickle
import re

def convert_to_windows_path(unix_path):
    # Replace forward slashes with backslashes
    path = re.sub(r"/", r"\\", unix_path)
    # Replace initial "/Users" with "C:\\Users"
    path = re.sub(r"^\\Users", r"C:\\Users", path)
    return path


def load_gz_p(file_name):
        print(file_name)  # KTS added this line
        with gzip.open(file_name, 'rb') as f:
            data = f.read()
        try:
            return pickle.loads(data, encoding='utf-8')
        except:
            return pickle.loads(data)
        
def load_old(f_name):
    song_data = []
   # print(f_name)  #KTS added this line
    with gzip.open(f_name, 'rb') as fin:
        for line in fin:
            json_line = json.loads(line, encoding='utf-8')
            song_data.append(json_line)
    return song_data 

class SongAnalysis(object):
    def __init__(self, directory, output_path=None, cores=None, json_out = None):

        if output_path is None:
            output_path = directory + "/AnalysisOutput_" + time.strftime("%Y%m%d_T%H%M%S")
        if cores is None:
            cores = mp.cpu_count()
        self.cores = cores
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        self.directory = directory
        
        if json_out is None:
            json_out = os.path.join(directory, "json_output")
        if not os.path.exists(json_out):
            os.makedirs(json_out)
        self.json_out = json_out
        os.makedirs(self.json_out, exist_ok=True)
        
        files = []
        file_names = []
        for f in os.listdir(directory):
            if f.endswith('gzip'):
                files.append(os.path.join(directory, f))    
                file_names.append(f)
        print(file_names)

        # this allows you to give the function directories that do not contain gzips without writing a file
        # if this is not in place, then a file will be created with FileName in the header but no other headers and
        # then will be used for same output path and headers will never be added.
        if len(files) == 0:
            print('return')
            return

        print("checkpoint A-2")
        processes = mp.Pool(cores, maxtasksperchild=1000)
        print("checkpoint A-1")
        final_output = processes.map(self.run_analysis, files)
        print("checkpoint A")
        self.output_bout_data(output_path, file_names, final_output)
        print("checkpoint B")
        super(SongAnalysis, self).__init__()
        print("checkpoint C")

    def export_to_json(self, file_name, onsets_seconds, offsets_seconds, bout_duration_seconds, ms_per_pixel):  # kts added 1/22/24
        try:
            print(f"Processing file: {file_name}")  # Debugging print

            # Extract relevant parts from file name
            parts = os.path.basename(file_name).split('_')
            if len(parts) < 3:
                raise ValueError(f"Unexpected file naming format: {file_name}")

            # Extract bird_id from "LgXX"
            lg_part = parts[1]  # Example: "Lg53"
            if not lg_part.startswith("Lg") or not lg_part[2:].isdigit():
                raise ValueError(f"Unexpected bird ID format: {lg_part}. Parts extracted: {parts}")

            bird_id = lg_part[2:]  # Extract numeric ID

            # Extract bout number from floating-point value
            try:
                bout_number = parts[2].split('.')[1]  # Extract decimal part only
            except IndexError:
                raise ValueError(f"Unexpected bout number format: {parts[2]}")

            # Check if the file is trimmed
            trimmed = file_name.endswith("trimmed.gzip")

            # Construct processed file name
            processed_file_name = f"SegSyllsOutput_Zebra-finch-{bird_id}_F1_{bout_number}{'_trimmed' if trimmed else ''}.json"
        
            # Determine output JSON path
            json_file_path = os.path.join(self.json_out, processed_file_name) if self.json_out else os.path.splitext(file_name)[0] + ".json"

            # Determine wav_location based on nesting format
            parent_dir = os.path.dirname(file_name)  # Get the directory of the input file
            wav_base_dir = os.path.join(os.path.dirname(parent_dir), "Wavs")  # Replace GZIPS with Wavs

            # Remove "SegSyllsOutput_" prefix if it exists
            file_base_name = os.path.basename(file_name).replace("SegSyllsOutput_", "")

            # Replace .gzip with .wav
            file_base_name = file_base_name.replace(".gzip", ".wav")

            # Check if the file includes "seewave_" in the name
            if "seewave_" in file_name:
                wav_location = os.path.join(wav_base_dir, "Edited_HPF_Amp", file_base_name)
            else:
                wav_location = os.path.join(wav_base_dir, "Raw", file_base_name)

            # Create JSON data
            json_data = {
                "length_s": bout_duration_seconds,
                "samplerate_hz": 44100,
                "wav_location": wav_location,
                "trimmed": trimmed,
                "indvs": {
                    bird_id: {
                        "species": "Zebra-finch",
                        "key": bout_number,
                        "units": {
                            "syllables": {
                                "start_times": onsets_seconds.tolist(),
                                "end_times": offsets_seconds.tolist(),
                            },
                        }
                    }
                }
            }

            print(f"Saving JSON to: {json_file_path}")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"JSON successfully saved to: {json_file_path}")
        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")  


    def run_analysis(self, filepath):
        # file names
        dirname, basename = os.path.split(filepath)
        print("checkpoint1")

        # load data
        self.onsets, self.offsets, self.threshold_sonogram, self.millisecondsPerPixel, self.hertzPerPixel, self.onsets_rescaled, self.offsets_rescaled, self.onsets_seconds, self.offsets_seconds = \
            self.load_bout_data(dirname, basename)
        print("checkpoint2")

        # run analysis
        syllable_durations, num_syllables, bout_stats, bout_duration_scaled, bout_duration_seconds = self.get_bout_stats()
        print("checkpoint3")
        syllable_stats = self.get_syllable_stats(syllable_durations, num_syllables)
        print("checkpoint4")
        note_stats = self.get_note_stats(num_syllables)

        # write output
        final_output = self.update_dict([bout_stats, syllable_stats, note_stats])
        self.export_to_json(os.path.join(dirname, basename), self.onsets_seconds, self.offsets_seconds, bout_duration_seconds, self.millisecondsPerPixel)
        
        print("checkpoint final_output")
        return final_output

    """ 
    Load sonogram and syllable marks (onsets and offsets).
    """
    
    
    def load_bout_data(self, dirname, basename):
        """
        Load sonogram and syllable marks (onsets and offsets).
        """
        f_name= os.path.join(dirname, basename)
        try:
            song_data = load_gz_p(f_name)
        except:
            song_data = load_old(f_name)
        
        onsets = np.asarray(song_data[1]['Onsets'], dtype='int')
        offsets = np.asarray(song_data[1]['Offsets'], dtype='int')
        threshold_sonogram = np.asarray(song_data[2]['Sonogram'])
        ms_per_pixel = np.asarray(song_data[3]['timeAxisConversion'])
        hz_per_pixel = np.asarray(song_data[3]['freqAxisConversion'])

        # for rescaling non-44100 sample rate files:
        # ms/pixel = 0.317460317 at 44100
        # hz/pixel = 42.98245614 at 44100
        onsets_rescaled = np.round(onsets*ms_per_pixel/0.317460317) # kts added 7/20
        offsets_rescaled = np.round(offsets*ms_per_pixel/0.317460317) # kts added 7/20

        onsets_seconds = onsets*ms_per_pixel/1000 # kts added 1/22/24
        offsets_seconds = offsets*ms_per_pixel/1000 # kts added 1/22/24

        #sono_name = os.path.join(dirname, basename.replace('.gzip', 'Sono.csv'))
        #np.savetxt(sono_name, threshold_sonogram, delimiter=',',  fmt='%d')


        # os.environ[SongStuff] = onsets, offsets, threshold_sonogram   #KTS added

        return onsets, offsets, threshold_sonogram, ms_per_pixel, hz_per_pixel, onsets_rescaled, offsets_rescaled, onsets_seconds, offsets_seconds

    """ 
    Write output.
    """
    def output_bout_data(self, output_path, file_name, output_dict):
        df_output = pd.DataFrame.from_dict(output_dict)
        df_output.index = [file_name]

        if not os.path.isfile(output_path + '.txt') and not os.path.isfile(output_path):
            if output_path.endswith('.txt'):
                df_output.to_csv(output_path, sep="\t", index_label='FileName')
            else:
                df_output.to_csv((output_path + '.txt'), sep="\t", index_label='FileName')
        else:
            if output_path.endswith('.txt'):
                df_output.to_csv(output_path, sep="\t", mode='a', header=False)
            else:
                df_output.to_csv(output_path + '.txt', sep="\t", mode='a', header=False)

    """
    General methods
    """
    def get_basic_stats(self, durations, data_type, units):
        if len(durations) == 0:  # just in case there is one syllable and so silence_durations is empty
            stats = {'largest_' + data_type + units: 'NA',
                     'smallest_' + data_type + units: 'NA',
                     'avg_' + data_type + units: 'NA',
                     'std_' + data_type + units: 'NA',
                     'All_Syllables_' + data_type + units: durations} 
                     # the All_Syllables variables correspond to the individual values for each syllable
        else:
            stats = {'largest_' + data_type + units: max(durations),
                     'smallest_' + data_type + units: min(durations),
                     'avg_' + data_type + units: np.mean(durations),
                     'std_' + data_type + units: np.std(durations, ddof=1),
                     'All_Syllables_' + data_type + units: durations,
                     'onsets': self.onsets, # kts
                     'offsets': self.offsets, # kts
                     'ms_per_pixel': self.millisecondsPerPixel,
                     'hz_per_pixel': self.hertzPerPixel,
                     'onsets_rescaled': self.onsets_rescaled,
                     'offsets_rescaled': self.offsets_rescaled}  # kts
                     # the All_Syllables variables correspond to the individual values for each syllable
        return stats

    def update_dict(self, dictionaries):
        new_dict = {}
        for d in dictionaries:
            new_dict.update(d)
        return new_dict

    def get_freq_stats(self, freq_range_upper, freq_range_lower, data_type):
        rows = np.shape(self.threshold_sonogram)[0]

        avg_upper_freq = np.mean(freq_range_upper)
        avg_lower_freq = np.mean(freq_range_lower) - 1  # must subtract 1 since freq_range_lower is exclusive
        # note: max freq is min row; min freq is max row --> python matrix indexes starting at 0 (highest freq)
        max_freq = min(freq_range_upper)
        min_freq = max(freq_range_lower) - 1  # must subtract 1 since freq_range_lower is exclusive
        overall_freq_range = abs(max_freq-min_freq) + 1  # add one back so difference is accurate (need min_freq
        # exclusive)

        avg_upper_freq_scaled = (rows-avg_upper_freq)*self.hertzPerPixel
        avg_lower_freq_scaled = (rows-avg_lower_freq)*self.hertzPerPixel
        max_freq_scaled = (rows-max_freq)*self.hertzPerPixel
        min_freq_scaled = (rows-min_freq)*self.hertzPerPixel
        overall_freq_range_scaled = overall_freq_range*self.hertzPerPixel

        freq_stats = {
                'avg_' + data_type + '_upper_freq(Hz)': avg_upper_freq_scaled,
                'avg_' + data_type + '_lower_freq(Hz)': avg_lower_freq_scaled,
                'max_' + data_type + '_freq(Hz)': max_freq_scaled,
                'min_' + data_type + '_freq(Hz)': min_freq_scaled,
                'overall_' + data_type + '_freq_range(Hz)': overall_freq_range_scaled,
                'All_Syllables_' + data_type + '_Upper_Freq_(Hz)': (rows-np.array(freq_range_upper))*self.hertzPerPixel,
                'All_Syllables_' + data_type + '_Lower_Freq_(Hz)': (rows-np.array(freq_range_lower))*self.hertzPerPixel}
        # the All_Syllables variables correspond to the individual values for each syllable

        freq_modulation_per_note = abs(np.asarray(freq_range_upper) - np.asarray(freq_range_lower))*self.hertzPerPixel
        basic_freq_stats = self.get_basic_stats(freq_modulation_per_note, data_type + '_freq_modulation', '(Hz)')

        freq_stats = self.update_dict([freq_stats, basic_freq_stats])

        return freq_stats

    """ 
    Analyze Bout: use onsets and offsets to get basic bout information (algebraic calcs)
    """
    def get_bout_stats(self):
        #print(self.onsets)
        #print(self.offsets)
        syllable_durations = self.offsets - self.onsets

        syllable_durations_scaled = syllable_durations*self.millisecondsPerPixel
        silence_durations_scaled = [self.onsets[i] - self.offsets[i-1] for i in range(1, len(self.onsets))]*self.millisecondsPerPixel
        bout_duration_scaled = (self.offsets[-1] - self.onsets[0])*self.millisecondsPerPixel
        bout_duration_seconds = bout_duration_scaled/1000

        num_syllables = len(syllable_durations)
        num_syllables_per_bout_duration = num_syllables/bout_duration_scaled

        song_stats = {'bout_duration(ms)': bout_duration_scaled,
                      'num_syllables': num_syllables,
                      'num_syllable_per_bout_duration(1/ms)': num_syllables_per_bout_duration} #, 'onsets': self.onsets,
                      # 'offsets': self.offsets}  # kts added onsets, offsets, then removed bc it's done in another section
        basic_syllable_stats = self.get_basic_stats(syllable_durations_scaled, 'syllable_duration', '(ms)')
        basic_silence_stats = self.get_basic_stats(silence_durations_scaled, 'silence_duration', '(ms)')
        bout_stats = self.update_dict([song_stats, basic_syllable_stats, basic_silence_stats])

        return syllable_durations, num_syllables, bout_stats, bout_duration_scaled, bout_duration_seconds

    """
    Analyze syllables: find unique syllables, syllable pattern, and stereotypy
    """
    def calc_sylls_freq_ranges(self):
        sylls_freq_range_upper = []
        sylls_freq_range_lower = []
        for j in range(len(self.onsets)):
            start = self.onsets[j]
            stop = self.offsets[j]
            rows_with_signal = np.nonzero(np.sum(self.threshold_sonogram[:, start:stop], axis=1))[0]
            sylls_freq_range_upper.append(rows_with_signal[0])  # will be inclusive
            sylls_freq_range_lower.append(rows_with_signal[-1] + 1)  # add one so that the lower will be exclusive

        return sylls_freq_range_upper, sylls_freq_range_lower

    def calc_max_correlation(self):
        sonogram_self_correlation = np.zeros(len(self.onsets))
        for j in range(len(self.onsets)):
            start = self.onsets[j]
            stop = self.offsets[j]
            sonogram_self_correlation[j] = (self.threshold_sonogram[:, start:stop]*self.threshold_sonogram[:,
                                                                                          start:stop]).sum()
        return sonogram_self_correlation

    def calc_syllable_correlation(self, a, b, shift_factor, min_length, max_overlap):
        syllable_correlation = []
        scale_factor = 100./max_overlap
        for m in range(shift_factor+1):
            syll_1 = self.threshold_sonogram[:, self.onsets[a]:(self.onsets[a] + min_length)]
            syll_2 = self.threshold_sonogram[:, (self.onsets[b] + m):(self.onsets[b] + min_length + m)]
            syllable_correlation.append(scale_factor*(syll_1*syll_2).sum())

        return syllable_correlation

    def get_sonogram_correlation(self, syllable_durations, corr_thresh=52.1):  #kts changed from 47.3
    #def get_sonogram_correlation(self, syllable_durations, corr_thresh=40):
        sonogram_self_correlation = self.calc_max_correlation()

        sonogram_correlation = np.zeros((len(self.onsets), len(self.onsets)))

        for j in range(len(self.onsets)):
            for k in range(len(self.onsets)):

                if j > k:  # do not want to fill the second half of the diagonal matrix
                    continue

                max_overlap = max(sonogram_self_correlation[j], sonogram_self_correlation[k])

                shift_factor = np.array(abs(syllable_durations[j]-syllable_durations[k]))

                if syllable_durations[j] < syllable_durations[k]:
                    min_length = syllable_durations[j]
                    syllable_correlation = self.calc_syllable_correlation(j, k, shift_factor, min_length, max_overlap)
                else:  # will be if k is shorter than j or they are equal
                    min_length = syllable_durations[k]
                    syllable_correlation = self.calc_syllable_correlation(k, j, shift_factor, min_length, max_overlap)

                # fill both upper and lower diagonal of symmetric matrix
                sonogram_correlation[j, k] = max(syllable_correlation)
                sonogram_correlation[k, j] = max(syllable_correlation)

        sonogram_correlation_binary = np.zeros(sonogram_correlation.shape)
        sonogram_correlation_binary[sonogram_correlation > corr_thresh] = 1  #kts print this

        # print(sonogram_correlation_binary)
        #print(np.shape(sonogram_correlation_binary))
        #print(len(sonogram_correlation_binary))
        return sonogram_correlation, sonogram_correlation_binary

    def find_syllable_pattern(self, sonogram_correlation_binary):
        # get syllable pattern
        syllable_pattern = np.zeros(len(sonogram_correlation_binary), 'int')
        for j in range(len(sonogram_correlation_binary)):
        #    print(syllable_pattern)
            syllable_pattern[j] = np.nonzero(sonogram_correlation_binary[:, j])[0][0]

        # check syllable pattern --> should be no new number that is smaller than it's index (ex: 12333634 --> the 4
        # should be a 3 but didn't match up enough; know this since 4 < pos(4) = 8)
        syllable_pattern_checked = np.zeros(syllable_pattern.shape, 'int')
        for j in range(len(syllable_pattern)):
            if syllable_pattern[j] < j:
                syllable_pattern_checked[j] = syllable_pattern[syllable_pattern[j]]
         #       print("this is syllable_pattern_checked (<j):")
         #       print(syllable_pattern_checked)
            else:
                syllable_pattern_checked[j] = syllable_pattern[j]
         #       print("this is syllable_pattern_checked (else):")
         #       print(syllable_pattern_checked)
        return syllable_pattern_checked


    def calc_syllable_stereotypy(self, sonogram_correlation, syllable_pattern_checked):
        syllable_stereotypy = np.zeros(len(sonogram_correlation))
        for j in range(len(sonogram_correlation)):
            x_syllable_locations = np.where(syllable_pattern_checked == j)[0]  # locations of all like syllables
            # initialize arrays
            x_syllable_correlations = np.zeros([len(syllable_pattern_checked), len(syllable_pattern_checked)])
            if len(x_syllable_locations) > 1:
                for k in range(len(x_syllable_locations)):
                    for h in range(len(x_syllable_locations)):
                        if k > h:  # fill only the lower triangle (not upper and not diagonal) so that similarities aren't double counted when taking the mean later
                            x_syllable_correlations[k, h] = sonogram_correlation[x_syllable_locations[k],
                                                                                 x_syllable_locations[h]]
            syllable_stereotypy[j] = np.nanmean(x_syllable_correlations[x_syllable_correlations != 0])

        return syllable_stereotypy

    def get_syllable_stats(self, syllable_durations, num_syllables, corr_thresh=47.3):  # corr_thresh was 40
        # get syllable correlations for entire sonogram
        sonogram_correlation, sonogram_correlation_binary = self.get_sonogram_correlation(syllable_durations,
                                                                                          corr_thresh)

        # get syllable pattern
        syllable_pattern_checked = self.find_syllable_pattern(sonogram_correlation_binary)

        # find unique syllables
        num_unique_syllables = len(np.unique(syllable_pattern_checked))
        num_syllables_per_num_unique = num_syllables / num_unique_syllables

        if num_syllables > 1:
            # determine how often the next syllable is the same as the previous syllable (for chippies, should be one
            # less than number of syllables in the bout)
            sequential_rep1 = len(np.where(np.diff(syllable_pattern_checked) == 0)[0])/(len(syllable_pattern_checked)-1)

            # determine syllable stereotypy
            syllable_stereotypy = self.calc_syllable_stereotypy(sonogram_correlation, syllable_pattern_checked)
            mean_syllable_stereotypy = np.nanmean(syllable_stereotypy)
            std_syllable_stereotypy = np.nanstd(syllable_stereotypy, ddof=1)
            syllable_stereotypy_final = syllable_stereotypy[~np.isnan(syllable_stereotypy)]
        else:
            sequential_rep1 = 'NA'
            syllable_stereotypy_final = 'NA'
            mean_syllable_stereotypy = 'NA'
            std_syllable_stereotypy = 'NA'

        syllable_stats_general = {'syll_correlation_threshold': corr_thresh,
                                  'num_unique_syllables': num_unique_syllables,
                                  'num_syllables_per_num_unique': num_syllables_per_num_unique,
                                  'syllable_pattern': syllable_pattern_checked.tolist(),
                                  'sequential_repetition': sequential_rep1,
                                  'syllable_stereotypy': syllable_stereotypy_final,
                                  'mean_syllable_stereotypy': mean_syllable_stereotypy,
                                  'std_syllable_stereotypy': std_syllable_stereotypy}

        sylls_freq_range_upper, sylls_freq_range_lower = self.calc_sylls_freq_ranges()
        syll_freq_stats = self.get_freq_stats(sylls_freq_range_upper, sylls_freq_range_lower, 'sylls')

        syllable_stats = self.update_dict([syllable_stats_general, syll_freq_stats])

        return syllable_stats

    """
    Analysis of notes: num of notes and categorization; also outputs freq ranges of each note
    """
    def get_notes(self):
        # zero anything before first onset or after last offset (not offset row is already zeros, so okay to include)
        # this will take care of any noise before or after the song before labeling the notes
        threshold_sonogram_crop = self.threshold_sonogram.copy()
        threshold_sonogram_crop[:, 0:self.onsets[0]] = 0
        threshold_sonogram_crop[:, self.offsets[-1]:-1] = 0

        labeled_sonogram, num_notes = label(threshold_sonogram_crop, return_num=True, connectivity=1)
                                                                        # ^connectivity 1=4 or 2=8(include diagonals)
        props = regionprops(labeled_sonogram)

        return num_notes, props

    def get_note_stats(self, num_syllables, note_size_thresh=25):
    #def get_note_stats(self, num_syllables, note_size_thresh=120):
        num_notes, props = self.get_notes()
        num_notes_updated = num_notes  # initialize, will be altered if the "note" is too small (<60 pixels)

        # stats per note
        notes_freq_range_upper = []
        notes_freq_range_lower = []
        note_length = []
        note_onset = []
        note_offset = []
        note_shape = []
        note_size = []

        # # note stats per bout
        num_flat = 0
        num_upsweeps = 0
        num_downsweeps = 0
        num_parabolas = 0

        for j in range(num_notes):
            note_ycoords = []  # clear/initialize for each note

            # use only the part of the matrix with the note
            sonogram_one_note = props[j].filled_image

            #print(np.size(sonogram_one_note))


            if np.size(sonogram_one_note) <= note_size_thresh:  # check the note is large enough to be a note and not
                # just noise
                note_length.append(0)  # place holder
                note_onset.append(0)
                note_offset.append(0)
                note_size.append(0)  # kts added
                #note_shape.append(0) # no place holder? seems ok
                num_notes_updated -= 1
            else:
                # use bounding box of the note (note, indexing is altered since python starts with 0 and we want to
                # convert rows to actual frequency)
                min_row, min_col, max_row, max_col = props[j].bbox
                notes_freq_range_upper.append(min_row)  # min row is inclusive (first row with labeled section)
                notes_freq_range_lower.append(max_row)  # max row is not inclusive (first zero row after the labeled section)

                note_length.append(np.shape(sonogram_one_note)[1])
                note_onset.append(min_col)
                note_offset.append(max_col)
                note_size.append(np.size(sonogram_one_note))

                #print(range(note_length[j]))

                for i in range(note_length[j]):
                    note_ycoords.append(np.mean(np.nonzero(sonogram_one_note[:, i])))  # [0])) # kts removed

                # kts un-indented the next 8 lines
                note_xcoords = np.arange(0, note_length[j])
                #print(i)
                #print(note_ycoords)
                #print(note_xcoords)
                poly = np.polyfit(note_xcoords, note_ycoords, deg=2)
                a = poly[0]
                b = poly[1]
                x_vertex = -b / (2 * a)  # gives x position of max or min of quadratic

                if np.isclose(a, 0, rtol=0.09, atol=0.015):  # check if the note is linear; kts changed from 9e-02
                    if np.isclose(b, 0, rtol=0.09, atol=0.09):
                        num_flat += 1
                        note_shape.append('flat')
                        # print('is flat')
                    elif b < 0:  # b is the slope if the poly is actually linear #kts flipped
                        num_upsweeps += 1
                        note_shape.append('upsweep')
                        # print('is upsweep')
                    else:
                        num_downsweeps += 1
                        note_shape.append('downsweep')
                        # print('is downsweep')
                    # now categorize non-linear notes
                elif x_vertex < .2 * note_length[j]:
                    if a < 0: # kts flipped 7/29/22
                        num_upsweeps += 1
                        note_shape.append('upsweep2')
                    else:
                        num_downsweeps += 1
                        note_shape.append('downsweep2')
                elif x_vertex > .8 * note_length[j]:
                    if a < 0: # kts flipped
                        num_downsweeps += 1
                        note_shape.append('downsweep3')
                    else:
                        num_upsweeps += 1
                        note_shape.append('upsweep3')
                else:  # the vertex is not within the first or last 20% of the note
                    num_parabolas += 1
                    note_shape.append('parabola')

    # collect stats into dictionaries for output
        note_length_array = np.asarray(note_length)
        note_length_array = note_length_array[note_length_array != 0]
        note_length_array_scaled = note_length_array*self.millisecondsPerPixel

        note_onset_array = np.asarray(note_onset)
        note_onset_array = note_onset_array[note_onset_array != 0]
        note_onset_array_scaled = np.round(note_onset_array * self.millisecondsPerPixel / 0.317460317)

        note_offset_array = np.asarray(note_offset)
        note_offset_array = note_offset_array[note_offset_array != 0]
        note_offset_array_scaled = np.round(note_offset_array * self.millisecondsPerPixel / 0.317460317)

        note_shape_array = np.asarray(note_shape)
        # note_shape_array = note_shape_array[note_shape_array != 0] kts didn't try this bc prob issues with char array, instead rm placeholder ~ ln461

        note_size_array = np.asarray(note_size)
        note_size_array = note_size_array[note_size_array != 0]

        note_counts = {'note_size_threshold': note_size_thresh,
                       'num_notes': num_notes_updated,
                       'num_notes_per_syll': num_notes_updated/num_syllables}
        basic_note_stats = self.get_basic_stats(note_length_array_scaled, 'note_duration', '(ms)')
        note_freq_stats = self.get_freq_stats(notes_freq_range_upper, notes_freq_range_lower, 'notes')

        note_categories = {'num_flat': num_flat, 'num_upsweeps': num_upsweeps,
                       'num_downsweeps': num_downsweeps, 'num_parabolas':
                           num_parabolas}

        note_arrays = {'note_lengths': note_length_array_scaled,
                       'note_onsets': note_onset_array_scaled,
                       'note_offsets': note_offset_array_scaled,
                       'note_onsets_unrescaled': note_onset_array,
                       'note_offsets_unrescaled': note_offset_array,
                       'note_shapes': note_shape_array,
                       'note_sizes': note_size_array}

        note_stats = self.update_dict([note_counts, basic_note_stats, note_freq_stats, note_categories, note_arrays])
        return note_stats


# part that actually calls/runs code  - BUT ALSO you need
#change directory to the folder containing the gzips to be analyzed
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Song sparrows from everywhere/R Analysis/Final Gzips/Sept2021_batch/SegSyllsOutput_20211020_T141330_KTS"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Song sparrows from everywhere/R Analysis/Gzips_testsubset/bout1_89675"
#directory = "/Users/kate/Box/Maria_Kate_Nicole/Song sparrows from everywhere/alex files for song sparrow/Test_subset/SegSyllsOutput_20211019_T134926"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/All The Sparrows/Gzips and R Analysis/Gzips/TestSubsetGzips/SmallerSubset"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/All The Sparrows/Chippering/Final Gzips"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Song sparrows from everywhere/R Analysis/Final Gzips/Sept2021_batch/SegSyllsOutput_20211020_T141330_KTS"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Gzip Files/Song Sparrow_KTS_Aggregated062022/Gzips_from_48000SampRate_wavs_copied"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Gzip Files/Dark Eyed Junco_KTS_Aggregated062022"
#directory = "/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Gzip Files/Song Sparrow_KTS_Aggregated062022" # sonogram output commented off - line 121
#directory = '/Users/kate/Documents/Creanza Lab/chipper_v1.0_osx/SongSparrowGzips'
#directory = '/Users/kate/Library/CloudStorage/Box-Box/Maria_Kate_Nicole/Gzip Files/Dark Eyed Junco_KTS_Aggregated062022' # sono output commented out
#directory = '/Users/boyerd/Library/CloudStorage/Box-Box/Creanza Lab/Rhythm_and_Complexity/RecordingsRanThroughChipper11:27/ClippedWavPlusGfiles11:27' # sono output commented out

# folders = [os.path.join(directory, f) for f in os.listdir(directory)]
if __name__ == '__main__':
    # for i in folders:
    #     print(i)
    #     SongAnalysis(1, i)
    SongAnalysis(1, directory)











