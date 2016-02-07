
import logging
import logging.config
import datetime
import numpy
from numpy import *
import scipy
import scipy.sparse
import scipy.io
import cPickle as pickle



import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt

from pegasos import pegasos
from pointsaga2 import pointsaga2
from pointsaga1 import pointsaga1
from saga import saga
from sdca import sdca
from csdca import csdca

import time


linestyles = ['-', '--', ':', '-.', '-', '--', '-.', ':',
              '-', '--', ':', '-.', '-', '--', '-.', ':',
              '-', '--', ':', '-.', '-', '--', '-.', ':']
colors = ['k', 'r', 'b', 'g', 'DarkOrange', 'm', 'y', 'c', 'Pink',
          '#00FF00', "#800000", "#808000", "#008080", "#B8860B",
          "#BDB76B", "#66CDAA", "#8A2BE2", "#8B4513"]

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger("opt")
    sTime = time.time()

    random.seed(42)

    def comparison(fname, default_props):
        
        datasetname = default_props.get("dataset", "rcv1_data.mat")

        logger.info("Loading data")
        dataset = scipy.io.loadmat(datasetname)
        X = dataset['X'].transpose()
        d = dataset['d'].flatten()
        
        n = X.shape[0]
        m = X.shape[1]
        logger.info("Data %s with %d points and %d features loaded.", datasetname, n, m)
        
        subset_dataset = default_props.get("subsetDataset", 1.0)
        if subset_dataset < 1:
            order = random.permutation(n)
            order = order[:ceil(n*subset_dataset)]
            X = X[order,:]
            d = d[order]

            n = X.shape[0]
            m = X.shape[1]
            logger.info("Subseted data to %d points (frac %1.4f)", n, subset_dataset)
        
        logger.info("Train Proportions: -1 %d   1: %d", sum(d == -1.0), sum(d == 1.0))
    
        # "Point-SAGA1", "Pegasos", 
        methods = ["Point-SAGA", "Pegasos", "SAGA", "SDCA", "Catalyst-SDCA"]
        
        baseline_method = "Point-SAGA"
        propsOverrides = {"Pegasos": {}, "Pegasos avg": {'returnAveraged': True}, 
                         "SAGA": {}, "Point-SAGA": {}, "Point-SAGA1": {}, "SDCA": {}, "Catalyst-SDCA": {}}
        stepSizes = {"Pegasos": [1024, 512, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05], 
                     "Pegasos avg": [1024, 512, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05], 
                     "SAGA": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
                     "Point-SAGA": [2056, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
                     "Point-SAGA1": [2056, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005],
                     "SDCA": [1],
                     "Catalyst-SDCA": [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001],
                 }
        runFunc = {"Pegasos": pegasos, "Pegasos avg": pegasos, "SAGA": saga, 
                   "Point-SAGA": pointsaga2, "Point-SAGA1": pointsaga1, 
                   "SDCA": sdca, "Catalyst-SDCA": csdca}
    
        #SDCA doesn't support 0 reg
        if default_props["reg"] == 0.0:
            del methods["SDCA"]
    
        fmins = {}
        fs = {}
        bestStepSize = {}
        
        for method in methods:
            fmins[method] = []
            fs[method] = {}
            for stepSize in stepSizes[method]:
                alt_props = {'stepSize': stepSize}
                props = default_props.copy()
                props.update(alt_props)
                props.update(propsOverrides[method])
                result = runFunc[method](X, d, props)
                fmins[method].append(result['flist'][-1])
                fs[method][stepSize] = result['flist']
              
            # Non-nan subset
            fmins_array = array(fmins[method])
            keep_flags = ~isnan(fmins_array)
            fmins_array = fmins_array[keep_flags]
            
            # Subset stepSizes as well with same indices
            nonnan_stepSizes = array(stepSizes[method])[keep_flags]
            
            best_idx = argmin(fmins_array)
            bestStepSize[method] = nonnan_stepSizes[best_idx]
            #import pdb; pdb.set_trace()
    
        print bestStepSize
        
        # Baseline fmin
        alt_props = {'stepSize':  bestStepSize[baseline_method], "passes": 100}
        props = default_props.copy()
        props.update(alt_props)
        result = runFunc[baseline_method](X, d, props)
        fstar = result['flist'][-1]
        
        plot_data = {'opts': {
            'methods': methods,
            'fstar': fstar, 
            'fs': fs,
            'props': default_props, 
            'stepSizes': stepSizes,
            'bestStepSize': bestStepSize,
            'fmins': fmins}}
        
        for method in methods:
            plot_data[method] = fs[method][bestStepSize[method]]
        
        #print(plot_data)
        pickle.dump(plot_data, open("%s.p" % fname, "wb"))
       
    def plot_all_stepsizes(fname): 
        
        plot_data = pickle.load(open("%s.p" % fname, "rb" )) 
        
        opts = plot_data['opts']
        del plot_data['opts']
        fstar = opts['fstar']
        
        for method in opts['methods']:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)

            ax1.set_ylabel("Function Suboptimality")
            ax1.set_xlabel("Epoch")
            ax1.set_yscale('log')
            
            captions = []
            idx = 0
            
            for reg in opts['stepSizes'][method]:
                flist = opts['fs'][method][reg]
                flist = [fi - fstar for fi in flist]

                ax1.plot(range(len(flist)), flist, linestyle=linestyles[idx], color=colors[idx])
                captions.append("$\gamma=%s$" % reg)
                idx = idx + 1

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                             box.width, box.height * 0.8])

            # Put a legend below current axis
            ax1.legend(captions, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, shadow=True, ncol=5)

            fname_method = "%s_allregs_%s" % (fname, method)
            logger.info("Saving plot: %s", fname_method)
            fig1.savefig("%s.pdf" % fname_method)
        
      
    def plot_comparison(fname):
        
        plot_data = pickle.load(open("%s.p" % fname, "rb" )) 
        
        opts = plot_data['opts']
        del plot_data['opts']
        fstar = opts['fstar']
        
        import matplotlib.pyplot as plt
        #plt.gca().tight_layout()
        fig1 = plt.figure(figsize=(4, 3))
        ax1 = fig1.add_subplot(111)

        ax1.set_ylabel("Function Suboptimality")
        ax1.set_xlabel("Epoch")
        ax1.set_yscale('log')
        
        captions = []
        idx = 0
        for method in opts['methods']:
            flist = copy(plot_data[method])
            flist = [fi - fstar for fi in flist]

            ax1.plot(range(len(flist)), flist, linestyle=linestyles[idx], color=colors[idx])
            if method == "SDCA":
                captions.append("%s" % (method))
            else:
                #captions.append("%s" % (method))
                captions.append("%s $\gamma=%s$" % (method, opts['bestStepSize'][method]))
            idx = idx + 1

        #fig1 = matplotlib.pyplot.gcf()
        #fig1.set_size_inches(4, 3)

        fig1.tight_layout()
        fig1.savefig("%s-nolegend.pdf" % fname)

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.15,
                         box.width, box.height * 0.8])

        # Put a legend below current axis
        ax1.legend(captions, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=False, 
                  labelspacing=0.1,
                  borderpad=0.3,
                  columnspacing=0.1,
                  fontsize="small",
                  ncol=2)
        
        logger.info("Saving plot: %s", fname)
        fig1.savefig("%s.pdf" % fname)
        
        # Put a legend below current axis
        ax1.legend(opts['methods'], loc='upper center', 
                  bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=False, 
                  labelspacing=0.1,
                  borderpad=0.3,
                  columnspacing=0.1,
                  fontsize="small",
                  ncol=2)
        
        fig1.savefig("%s_plain.pdf" % fname)
        

    def comparisonSpread(basename, default_props, binary=True):
        regs = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.0]
        
        if binary:
            lossnames = ["logistic", "hinge"]
        else:
            lossnames = ["squared"]
        
        for reg in regs:
            for lossname in lossnames:
                regpart = str(reg).split('.')[-1]
                fn = "%s_%s_reg_%s" % (basename, lossname, regpart)
                
                props = default_props.copy()
                props.update({'loss': lossname, "reg": reg})
                
                comparison(fn, props)
                plot_comparison(fn)
                
    def datasetSizeSpread(basename, default_props, binary=True, hinge=False):
        sizes = [1.0, 0.1, 0.05]
        
        if binary:
            if hinge:
                lossnames = ["hinge"]
            else:
                lossnames = ["logistic"]
        else:
            lossnames = ["squared"]
        
        for subset_size in sizes:
            for lossname in lossnames:
                sizepart = str(int(100*subset_size))
                regpart = str(default_props['reg']).split('.')[-1]
                fn = "%s_%s_reg_%s_size_%sp" % (basename, lossname, regpart, sizepart)
                
                props = default_props.copy()
                props.update({'loss': lossname, "subsetDataset": subset_size})
                
                comparison(fn, props)
                plot_comparison(fn)

    fn = "plots/australian"
    datasetSizeSpread(fn, {'reg': 0.0001, 'dataset': 'australian_scale.mat', 'passes': 30})

    fn = "plots/mushrooms"
    datasetSizeSpread(fn, {'reg': 0.0001, 'dataset': 'mushrooms.mat', 'passes': 30})
    
    fn = "plots/covtype"
    datasetSizeSpread(fn, {'reg': 0.000002, 'dataset': 'covtype.libsvm.binary.scale.mat', 'passes': 20})

    fn = "plots/rcv1"
    datasetSizeSpread(fn, {'reg': 5e-5, 'dataset': 'rcv1_train.binary.mat', 'passes': 40}, hinge=True)
