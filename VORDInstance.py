
class VORDInstance:

    def __init__(self, video_id, video_path, frame_count, fps, width, height,
                 subject_objects, trajectories, relation_instances):
        self.video_id = video_id
        self.video_path = video_path
        self.frame_count = frame_count
        self.fps = fps
        self.height = height
        self.width = width
        self.subject_objects = subject_objects
        self.trajectories = trajectories
        self.relation_instances = relation_instances

    def __repr__(self):
        return "VORD Instance: video_id=" + str(self.video_id)

    def include_object(self, object_label):
        for each_so in self.subject_objects:
            if each_so['category'].lower() == object_label.lower():
                return True
        return False

    def get_object_trajs(self, object_label):
        if self.include_object(object_label):
            trajs_list = []
            for each_so in self.subject_objects:
                if object_label == each_so['category']:
                    obj_tid = each_so['tid']
                    for each_traj in self.trajectories:
                        for each_traj_obj in each_traj:
                            if obj_tid == each_traj_obj['tid']:
                                trajs_list.append(each_traj_obj)
            return trajs_list
        else:
            return None

    def get_object_relations_list(self):
        objects_list = []
        relations_list = []
        for each_so in self.subject_objects:
            objects_list.append(each_so['category'])

        for each_rel in self.relation_instances:
            relations_list.append(each_rel['predicate'])
        # print("Video " + str(self.video_id) + " has "
        #       + str(len(objects_list)) + " objects and " +
        #       str(len(relations_list)) + " relations.")
        return objects_list, relations_list

    def get_triplet_list(self):
        categorys = {}
        for each_os in self.subject_objects:
            categorys[each_os['tid']] = each_os['category']

        triplet_list = []
        for each_pred in self.relation_instances:
            each_trip = (categorys[each_pred['subject_tid']],
                         each_pred['predicate'],
                         categorys[each_pred['object_tid']])
            triplet_list.append(each_trip)

        return triplet_list
