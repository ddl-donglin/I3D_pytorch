import json


def show_charades():
    with open('charades.json', 'r') as in_f:
        charades_json = json.load(in_f)

        print(charades_json.keys())
        print(len(charades_json.keys()))            # 9848
        print(charades_json['7UPGT'].keys())        # dict_keys(['subset', 'duration', 'actions'])
        print(len(charades_json['7UPGT'].keys()))   # 3
        print(charades_json['7UPGT']['subset'])     # training
        print(charades_json['7UPGT']['duration'])   # each_video_length (seconds)
        print(charades_json['7UPGT']['actions'])    # [action_id, action_begin_second, action_end_second]
        # sum_duration = 0
        # for each_at in charades_json['7UPGT']['actions']:
        #     sum_duration += each_at[2] - each_at[1]
        # print(sum_duration)


if __name__ == '__main__':
    show_charades()
