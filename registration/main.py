from utils import Datahandler
from utils import Preprocess
from utils import Networks
import voxelmorph as vxm
import argparse
import sys

case = ''
preprocess = ''
retrain_path = ''
loss = ''

# Read in arguments
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case', required=True, default='testing',
                        help="choose a case {intramodal T1w (=t1), intramodal T2 (=t2), "
                             "intermodal T1w-T2w(t1t2), test mode (testing)}",
                        type=str, choices=['t1', 't2', 't1t2', 'testing'])

    parser.add_argument('-pp', '--preprocessing', required=False, default=False, help="enable additional preprocessing",
                        type=bool, choices=[True, False])

    parser.add_argument('-rp', '--retrain_path', required=False, default=None, help="path to previous trained model",
                        type=str)

    parser.add_argument('-l', '--regularization_loss', required=False, default='l2',
                        help="define regularization loss {'l2', 'l1'}", choices=['l2', 'l1'])

    args = parser.parse_args()

    case = args.case
    preprocess = args.preprocessing
    retrain_path = args.retrain_path
    loss = args.regularization_loss

except KeyError:
    e = sys.exc_info()[0]
    print(e)

if case == 't1':

    if retrain_path is None:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from scratch")
    else:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from pretrained model")

    dh_t1 = Datahandler('brain_t1')

    if preprocess:
        pp_t1 = Preprocess(dh_t1)
        pp_t1.preprocess()

    if loss == 'l2':
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    else:
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l1').loss]

    loss_weights = [1, 0.01]

    nw_t1 = Networks(dh_t1, losses, loss_weights)

    if retrain_path is None:
        nw_t1.train_vxm()
    else:
        nw_t1.train_from_weights_vxm(retrain_path)


elif case == 't2':

    if retrain_path is None:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from scratch")
    else:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from pretrained model")

    dh_t2 = Datahandler('brain_t2')

    if preprocess:
        pp_t2 = Preprocess(dh_t2)
        pp_t2.preprocess()

    if loss == 'l2':
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    else:
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l1').loss]

    loss_weights = [1, 0.01]

    nw_t2 = Networks(dh_t2, losses, loss_weights)

    if retrain_path is None:
        nw_t2.train_vxm()
    else:
        nw_t2.train_from_weights_vxm(retrain_path)


elif case == 't1t2':

    if retrain_path is None:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from scratch")
    else:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from pretrained model")

    dh_t1t2 = Datahandler('inter_modal_t1t2')

    if preprocess:
        pp_t1 = Preprocess(dh_t1t2)
        pp_t1.preprocess()

        pp_t2 = Preprocess(dh_t1t2)
        pp_t2.preprocess()

    if loss == 'l2':
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    else:
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l1').loss]

    loss_weights = [1, 0.01]

    nw_t1t2 = Networks(dh_t1t2, losses, loss_weights)

    if retrain_path is None:
        nw_t1t2.train_vxm()
    else:
        nw_t1t2.train_from_weights_vxm(retrain_path)

else:

    if retrain_path is None:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from scratch")
    else:
        print("main for: " + case + " with preprocessing = " + str(preprocess) + " training from pretrained model")

    dh = Datahandler('testing')

    if loss == 'l2':
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    else:
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l1').loss]

    loss_weights = [1, 0.01]

    nh = Networks(dh, losses, loss_weights)

    if retrain_path is None:
        nh.train_vxm()
    else:
        nh.train_from_weights_vxm(retrain_path)

    nb_pairs = 2

    test_input, test_output = nh.predict_vxm(nb_pairs)

    nh.evaluate_axes_vxm(test_input, test_output)

    nh.evaluate_displ_vxm(test_input, test_output)

    nh.evaluate_losses_vxm(test_input, test_output)

    nh.evaluate_loss_history(metric='eval')
