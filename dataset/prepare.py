import json
import os


def gen_pred_dict(anno_rpath):
    pred_set = set()
    obj_set = set()
    for each_anno in os.listdir(anno_rpath):
        with open(os.path.join(anno_rpath, each_anno), 'r') as in_f:
            each_anno_json = json.load(in_f)

        for each_ins in each_anno_json['relation_instances']:
            pred_set.add(each_ins['predicate'])

        for each_objs in each_anno_json['subject/objects']:
            obj_set.add(each_objs['category'])

    pred_dict = dict()
    obj_dict = dict()
    for idx, pred in enumerate(pred_set):
        pred_dict[pred] = idx

    for idx, obj in enumerate(obj_set):
        obj_dict[obj] = idx

    with open('vidvrd_pred.json', 'w+') as out_f:
        out_f.write(json.dumps(pred_dict))

    with open('vidvrd_objs.json', 'w+') as out_f:
        out_f.write(json.dumps(obj_dict))


def gen_actions_dict(pred_dict='vidvrd_pred.json'):
    with open(pred_dict, 'r') as in_f:
        preds = json.load(in_f)

    first_set = set()
    second_set = set()
    third_set = set()
    for each_pred in preds.items():
        each_pred_splits = each_pred[0].split('_')
        if len(each_pred_splits) == 1:
            first_set.add(each_pred_splits[0])
        if len(each_pred_splits) == 2:
            second_set.add(each_pred_splits[1])
        if len(each_pred_splits) == 3:
            third_set.add(each_pred_splits[2])

    print(first_set)
    print(second_set)
    print(third_set)

    action_dict = dict()
    rela_dict = dict()
    for idx, act in enumerate(first_set - second_set):
        action_dict[act] = idx

    for idx, rela in enumerate(second_set):
        rela_dict[rela] = idx

    with open('actions.json', 'w+') as out_f:
        out_f.write(json.dumps(action_dict))

    with open('spatio_rela.json', 'w+') as out_f:
        out_f.write(json.dumps(rela_dict))


if __name__ == '__main__':
    # gen_pred_dict('/home/daivd/PycharmProjects/VidVRD-dataset/train_anno')
    gen_actions_dict()
