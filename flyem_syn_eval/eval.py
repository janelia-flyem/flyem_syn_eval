from libdvid import DVIDNodeService, ConnectionMethod
import pulp
import numpy as np
from collections import namedtuple
import json, os

class Tbar_Info(namedtuple('Tbar_Info',
                           ['annot','server','uuid','roi'])):
    __slots__ = ()
    def __new__(cls, annot, server=None, uuid=None, roi=None):
        return super(Tbar_Info, cls).__new__(
            cls, annot, server, uuid, roi)
    def get_tbars(self, conf_filter=None):
        if os.path.isfile(self.annot): # read from json
            tbars_out = tbars_from_json(self.annot, conf_filter)
        else: # get from dvid
            tbars_out = tbars_from_dvid(self.server, self.uuid,
                                        self.annot, self.roi,
                                        conf_filter)
        return tbars_out
Tbars = namedtuple('Tbar', 'pos conf')
PR_Result = namedtuple(
    'PR_Result', 'num_tp tot_pred tot_gt pp rr match')

def cx_synapse_groundtruth(roi_suffix):
    return Tbar_Info('combined_synapses_08302016',
                     'emdata2:8000', 'cb7dc',
                     'roi_cx1_%s' % roi_suffix)

def evaluate_pr(tbars_pd, tbars_gt,
                dist_thresh=27, conf_thresholds=None):

    tbars_pd = tbars_pd.get_tbars()
    tbars_gt = tbars_gt.get_tbars(1.)

    if conf_thresholds is None:
        cc = np.unique(tbars_pd.conf)
        conf_thresholds = (cc[1:]+cc[:-1])/2

    result = obj_pr_curve(tbars_pd, tbars_gt,
                          dist_thresh, conf_thresholds)
    return result


def get_synapses_dvid(dvid_server, dvid_uuid, dvid_annot, dvid_roi):
    dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                'flyem_syn_eval','flyem_syn_eval')
    synapses_json = dvid_node.custom_request(
        '%s/roi/%s' % (dvid_annot, dvid_roi),
        None, ConnectionMethod.GET)
    return synapses_json

def tbars_from_dvid(dvid_server, dvid_uuid, dvid_annot, dvid_roi,
                    conf_filter=None):
    synapses_json = get_synapses_dvid(
        dvid_server, dvid_uuid, dvid_annot, dvid_roi)
    return tbars_from_json(synapses_json, conf_filter)

def tbars_from_json(synapses_json, conf_filter=None):
    if isinstance(synapses_json, str):
        if os.path.isfile(synapses_json):
            with open(synapses_json) as json_file:
                synapses = json.load(json_file)
        else:
            synapses = json.loads(synapses_json)
    else:
        synapses = synapses_json

    pos     = []
    conf    = []
    for synapse in synapses:
        if synapse['Kind'] != 'PreSyn':
            continue

        if synapse['Prop'].has_key('conf'):
            cc = float(synapse['Prop']['conf'])
        else:
            cc = 1.0

        if conf_filter is not None:
            if cc < conf_filter - 1e-4:
                continue

        pos.append(synapse['Pos'])
        conf.append(cc)

    return Tbars(np.asarray(pos, dtype='float32'),
                 np.asarray(conf, dtype='float32'))


def obj_pr_curve(predict, groundtruth, dist_thresh, thresholds,
                 allow_mult=False):

    predict_locs     = predict.pos
    predict_conf     = predict.conf
    groundtruth_locs = groundtruth.pos

    n_thd    = thresholds.size
    num_tp   = np.zeros( (n_thd,) )
    tot_pred = np.zeros( (n_thd,) )
    tot_gt   = np.zeros( (n_thd,) )
    pp       = np.zeros( (n_thd,) )
    rr       = np.zeros( (n_thd,) )

    for ii in range(thresholds.size):
        predict_locs_iter = predict_locs[
            predict_conf >= thresholds[ii],:]
        mm = obj_pr(predict_locs_iter, groundtruth_locs,
                    dist_thresh, allow_mult=allow_mult)
        num_tp[  ii] = mm.num_tp
        tot_pred[ii] = mm.tot_pred
        tot_gt[  ii] = mm.tot_gt
        pp[      ii] = mm.pp
        rr[      ii] = mm.rr

    return PR_Result(num_tp=num_tp, tot_pred=tot_pred, tot_gt=tot_gt,
                     pp=pp, rr=rr, match=None)

def obj_pr(predict_locs, groundtruth_locs, dist_thresh,
           allow_mult=False):

    if( (predict_locs.shape[0] == 0) |
        (groundtruth_locs.shape[0] == 0) ): # check for empty cases
        tot_pred = predict_locs.shape[0]
        tot_gt   = groundtruth_locs.shape[0]

        pp = 0
        rr = 0
        if(tot_pred == 0):
            pp = 1
        if(tot_gt   == 0):
            rr = 1

        return PR_Result(num_tp=0, tot_pred=tot_pred, tot_gt=tot_gt,
                         pp=pp, rr=rr, match=None)

    pred     = predict_locs.reshape(     (-1, 1,3) )
    gt       = groundtruth_locs.reshape( ( 1,-1,3) )

    dists    = np.sqrt( ((pred-gt)**2).sum(axis=2) )
    dists   -= dist_thresh

    match    = obj_match(dists, allow_mult=allow_mult)

    num_tp   = float(match.sum())
    pd_mult  = np.maximum(match.sum(axis=1) - 1,0).sum()
    tot_pred = match.shape[0] + pd_mult

    result   = PR_Result(
        num_tp=num_tp, tot_pred=tot_pred, tot_gt=match.shape[1],
        pp=num_tp/match.shape[0], rr=num_tp/match.shape[1],
        match=match)

    return result

def obj_match(dists, allow_mult=False):
    """compute object matching

    sets up and solves an integer program for matching predicted
    object locations to ground-truth object locations
    """

    match = pulp.LpProblem('obj match', pulp.LpMinimize)

    n_pred, n_gt = dists.shape

    use_var = (dists < 0)

    match_names  = []
    match_costs = {}
    match_const_pred = [ list() for ii in range(n_pred)]
    match_const_gt   = [ list() for ii in range(n_gt)]

    for ii in range(n_pred):
        for jj in range(n_gt):
            if use_var[ii,jj]:
                match_names.append('x_%d_%d' % (ii, jj))
                match_costs[match_names[-1]] = dists[ii,jj]
                match_const_pred[ii].append(match_names[-1])
                match_const_gt[  jj].append(match_names[-1])

    match_vars = pulp.LpVariable.dicts(
        'var', match_names, 0, 1, pulp.LpInteger)
    match += pulp.lpSum([match_costs[ii]*match_vars[ii] for
                         ii in match_names]), 'obj match cost'

    if not allow_mult:
        for ii in range(n_pred):
            if match_const_pred[ii]:
                match += pulp.lpSum(match_vars[jj] for
                                    jj in match_const_pred[ii]
                ) <= 1, 'pred %d' % ii

    for ii in range(n_gt):
        if match_const_gt[ii]:
            match += pulp.lpSum(match_vars[jj] for
                                jj in match_const_gt[ii]
            ) <= 1, 'gt %d' % ii

    match.solve()

    obj_matches = np.zeros( (n_pred,n_gt), dtype='bool')

    for ii in range(n_pred):
        for jj in range(n_gt):
            if use_var[ii,jj]:
                obj_matches[ii,jj] = match_vars[
                    'x_%d_%d' % (ii, jj)].varValue

    return obj_matches
