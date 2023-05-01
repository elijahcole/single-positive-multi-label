import json
import numpy as np
import matplotlib.pyplot as plt

cat_name_to_weights = None
cat_id_to_cat_name = None

def get_anns_by_imID():
    imID_to_anns = {}
    for phase in ['val']:
        f_val = open('instances_val2014.json')
        D = json.load(f_val)
        for i in range(len(D['annotations'])):
            if not D['annotations'][i]['image_id'] in imID_to_anns:
                imID_to_anns[D['annotations'][i]['image_id']] = []
            imID_to_anns[D['annotations'][i]['image_id']].append(D['annotations'][i])
        f_val.close()
    return imID_to_anns


def get_dicts(annotations):
    cat_to_imgs = {}
    im_id_to_anns = {}
    for annotation in annotations:
        im_id = annotation['im_id']
        cat_id = annotation['cat_id']
        if cat_id not in cat_to_imgs:
            cat_to_imgs[cat_id] = []
        cat_to_imgs[cat_id].append(im_id)
        if im_id not in im_id_to_anns:
            im_id_to_anns[im_id] = []
        im_id_to_anns[im_id].append(annotation)
    return (cat_to_imgs, im_id_to_anns)


def get_bboxes_by_imID(coco):
    imID_to_cat_to_bboxes = {}
    for imID in coco:
        if imID not in imID_to_cat_to_bboxes:
            imID_to_cat_to_bboxes[imID] = {}
        for ann in coco[imID]:
            cat = ann['category_id']
            if cat not in imID_to_cat_to_bboxes[imID]:
                imID_to_cat_to_bboxes[imID][cat] = []
            imID_to_cat_to_bboxes[imID][cat].append(ann['bbox'])
    return imID_to_cat_to_bboxes


def get_imID_to_dims():
    imID_to_dims = {}
    for phase in ['val']:
        f = open('instances_{}2014.json'.format(phase))
        D = json.load(f)
        for i in range(len(D['images'])):
            imID_to_dims[D['images'][i]['id']] = [D['images'][i]['width'], D['images'][i]['height']]
        f.close()
    return imID_to_dims


def point_in_bbox(x, y, im_id, cat, imID_to_cat_to_bboxes):
    bboxes = imID_to_cat_to_bboxes[im_id][cat]
    for bbox in bboxes:
        if x >= bbox[0] and x <= bbox[2] and y >= bbox[1] and y <= bbox[2]:
            return True
    return False


def get_cat_id_to_cat_name():
    cat_list_f = open('categories.json')
    cat_list = json.load(cat_list_f)
    cat_id_2_cat_name = {}  # map: cat_id --> cat_name
    cat_name_2_cat_id = {}
    for entry in cat_list['categories']:
        cat_id_2_cat_name[entry['id']] = entry['name']
        cat_name_2_cat_id[entry['name']] = entry['id']
    cat_list_f.close()
    return cat_id_2_cat_name


def compare_files():
    f1 = open('crowdsourcing_instance_spotting_val2014.json')
    annotations1 = json.load(f1)
    imids1 = set()
    for ann in annotations1:
        imids1.add(ann['im_id'])

    f2 = open('crowdsourcing_annotate_category_val2014.json')
    annotations2 = json.load(f2)
    imids2 = set()
    for ann in annotations2:
        imids2.add(ann['im_id'])
    
    overlap = imids1 & imids2
    print('number of shared imids {}'.format(len(overlap)))
    print(f'instance has {len(imids1)} while annotate has {len(imids2)} unique image ids')

    f3 = open('instances_val2014.json')
    D = json.load(f3)
    imids3 = set()
    for i in range(len(D['annotations'])):
        imids3.add(D['annotations'][i]['image_id'])
    
    diff1 = imids1 - imids3
    print(f'There are {len(diff1)} image ids found in raw instance but not COCO instance')
    diff2 = imids2 - imids3
    print(f'There are {len(diff2)} image ids found in raw annotate but not COCO instance')
    f1.close()
    f2.close()
    f3.close()


# if __name__ == '__main__':
mode = 'run'  # 'run', or 'check'

cat_weights = {}
if mode == 'run':
    coco_by_imID = get_anns_by_imID()
    imID_to_cat_to_bboxes = get_bboxes_by_imID(coco_by_imID)
    im_id_to_dims = get_imID_to_dims()
    cat_id_to_cat_name = get_cat_id_to_cat_name()
    avg_weights = np.zeros((len(cat_id_to_cat_name),))
    count = 0
    for f in ['crowdsourcing_annotate_category_val2014.json',
                'crowdsourcing_instance_spotting_val2014.json']:
        f = open(f)
        annotations = json.load(f)
        (cat_to_imgs, im_id_to_anns) = get_dicts(annotations)
        cat_to_prob = {}
        iterations = 0
        cat_to_pts = {}
        for cat in cat_to_imgs:
            good_clicks = 0
            total_clicks = 0
            cat_to_pts[cat] = []
            for im_id in cat_to_imgs[cat]:
                for ann in im_id_to_anns[im_id]:
                    total_clicks += 1
                    cat_id = ann['cat_id']
                    if not cat_id == cat:
                        continue
                    x = im_id_to_dims[im_id][0] * ann['x']
                    y = im_id_to_dims[im_id][1] * ann['y']
                    try:
                        if point_in_bbox(x, y, im_id, cat_id, imID_to_cat_to_bboxes):
                            good_clicks += 1
                            cat_to_pts[cat].append((ann['x'], ann['y']))
                    except KeyError:
                        iterations += 1
                        continue
            cat_to_prob[cat] = good_clicks/total_clicks
        cat_name_list = [cat_id_to_cat_name[cat] for cat in cat_to_prob]
        # MAKE HEAT MAP
        # for c_id in cat_to_pts:
        #     # get list of x,y pts
        #     x_pts = []
        #     y_pts = []
        #     for pt in cat_to_pts[c_id]:
        #         x_pts.append(pt[0])
        #         y_pts.append(pt[1])
        #     plt.scatter(x_pts, y_pts)
        #     cat_name = cat_id_to_cat_name[c_id]
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # fig = plt.gcf()
            # fig.set_size_inches(10, 10)
            # if count == 0:
            #     plt.title(cat_name + '_ann')
            #     plt.savefig(f'plots/{cat_name}_ann')
            # else:
            #     plt.title(cat_name + '_inst')
            #     plt.savefig(f'plots/{cat_name}_inst')
            # plt.clf()
        vals = list(cat_to_prob.values())
        # sort them by name
        combined = []
        for r in range(len(vals)):
            combined.append((cat_name_list[r], vals[r]))
        combined.sort(key=lambda y: y[0])
        # get them back
        cat_name_list = []
        vals = []
        for l in range(len(combined)):
            cat_name_list.append(combined[l][0])
            vals.append(combined[l][1])
            avg_weights[l] += combined[l][1]
        # plt.bar(cat_name_list, vals)
        # plt.xticks(rotation = 90)
        # fig = plt.gcf()
        # fig.set_size_inches(24, 10.5)
        # plt.title(f)
        # fig.savefig(f'prob_model{count}.png', dpi=100)
        # plt.clf()
        count += 1
        f.close()
    avg_weights *= 0.5
    cat_name_list = [cat_id_to_cat_name[cat] for cat in cat_to_prob]
    cat_name_list.sort()
    zip_iterator = zip(cat_name_list, avg_weights)
    cat_name_to_weights = dict(zip_iterator)  # this is used in another module to sample
    # plt.bar(cat_name_list, avg_weights)
    # plt.xticks(rotation = 90)
    # fig = plt.gcf()
    # fig.set_size_inches(24, 10.5)
    # plt.title('averaged_weights')
    # fig.savefig('empirical_distr.png', dpi=100)
    # plt.clf()
elif mode == 'check':
    compare_files()
