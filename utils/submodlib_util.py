from statistics import mode
import numpy as np
from submodlib import FacilityLocationMutualInformationFunction
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import ConcaveOverModularFunction
from submodlib_cpp import ConcaveOverModular
from submodlib import GraphCutMutualInformationFunction
from submodlib import LogDeterminantMutualInformationFunction
from submodlib import FacilityLocationConditionalMutualInformationFunction

def Random_wrapper(image_list, budget=10):
    rand_idx = np.random.permutation(len(image_list))[:budget]
    rand_idx = rand_idx.tolist()
    Random_results = [image_list[i] for i in rand_idx]

    return Random_results


def FL1MI_wrapper(lake_data,
                  query_data,
                  eta,
                  image_list,
                  budget=10,
                  metric="cosine",
                  optimizer="NaiveGreedy",
                  stopIfZeroGain=False,
                  stopIfNegativeGain=False,
                  verbose=False):
    # print(lake_data.shape)
    # print(query_data)
    obj = FacilityLocationMutualInformationFunction(
        n=lake_data.shape[0],
        num_queries=query_data.shape[0],
        data=lake_data,
        queryData=query_data,
        metric=metric,
        magnificationEta=eta)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def FL2MI_wrapper(lake_data,
                  query_data,
                  eta,
                  image_list,
                  budget=10,
                  metric="cosine",
                  optimizer="NaiveGreedy",
                  stopIfZeroGain=False,
                  stopIfNegativeGain=False,
                  verbose=False):
    obj = FacilityLocationVariantMutualInformationFunction(
        n=lake_data.shape[0],
        num_queries=query_data.shape[0],
        data=lake_data,
        queryData=query_data,
        metric=metric,
        queryDiversityEta=eta)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def FL2MI_wrapper_kernel(query_data,
                         eta,
                         budget=10,
                         metric="cosine",
                         optimizer="NaiveGreedy",
                         stopIfZeroGain=False,
                         stopIfNegativeGain=False,
                         verbose=False):
    obj = FacilityLocationVariantMutualInformationFunction(
        n=query_data.shape[1],
        num_queries=query_data.shape[0],
        query_sijs=query_data.T,
        queryDiversityEta=eta)

    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def COM_wrapper(lake_data,
                query_data,
                eta,
                image_list,
                budget=10,
                metric="cosine",
                optimizer="LazierThanLazyGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False):
    obj = ConcaveOverModularFunction(n=lake_data.shape[0],
                                     num_queries=query_data.shape[0],
                                     data=lake_data,
                                     queryData=query_data,
                                     metric=metric,
                                     queryDiversityEta=eta,
                                     mode=ConcaveOverModular.logarithmic)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def COM_wrapper_kernel(query_data,
                       eta,
                       budget=10,
                       metric="cosine",
                       optimizer="NaiveGreedy",
                       stopIfZeroGain=False,
                       stopIfNegativeGain=False,
                       verbose=False):
    obj = ConcaveOverModularFunction(n=query_data.shape[1],
                                     num_queries=query_data.shape[0],
                                     query_sijs=query_data.T,
                                     queryDiversityEta=eta)

    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def GCMI_wrapper(lake_data,
                 query_data,
                 eta,
                 image_list,
                 budget=10,
                 metric="cosine",
                 optimizer="LazierThanLazyGreedy",
                 stopIfZeroGain=False,
                 stopIfNegativeGain=False,
                 verbose=False):
    obj = GraphCutMutualInformationFunction(n=lake_data.shape[0],
                                            num_queries=query_data.shape[0],
                                            data=lake_data,
                                            queryData=query_data,
                                            metric=metric)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]

def GCMI_wrapper_kernel(query_data,
                       eta,
                       budget=10,
                       metric="cosine",
                       optimizer="NaiveGreedy",
                       stopIfZeroGain=False,
                       stopIfNegativeGain=False,
                       verbose=False):
    obj = GraphCutMutualInformationFunction(n=query_data.shape[1],
                                     num_queries=query_data.shape[0],
                                     query_sijs=query_data.T,
                                     metric=metric)

    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]

def LogDetMI_wrapper(lake_data,
                     query_data,
                     eta,
                     image_list,
                     budget=10,
                     metric="cosine",
                     optimizer="LazierThanLazyGreedy",
                     stopIfZeroGain=False,
                     stopIfNegativeGain=False,
                     verbose=False,
                     lambdaVal=1):
    obj = LogDeterminantMutualInformationFunction(
        n=lake_data.shape[0],
        num_queries=query_data.shape[0],
        data=lake_data,
        queryData=query_data,
        metric=metric,
        magnificationEta=eta,
        lambdaVal=lambdaVal)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def FL1CMI_wrapper(lake_data,
                   query_data,
                   private_data,
                   eta,
                   image_list,
                   budget=10,
                   metric="cosine",
                   optimizer="LazierThanLazyGreedy",
                   stopIfZeroGain=False,
                   stopIfNegativeGain=False,
                   verbose=False,
                   lambdaVal=1):
    obj = FacilityLocationConditionalMutualInformationFunction(
        n=lake_data.shape[0],
        num_queries=query_data.shape[0],
        data=lake_data,
        num_privates=private_data.shape[0],
        queryData=query_data,
        privateData=private_data,
        metric=metric,
        magnificationEta=eta)
    greedyList = obj.maximize(budget=budget,
                              optimizer=optimizer,
                              stopIfZeroGain=stopIfZeroGain,
                              stopIfNegativeGain=stopIfNegativeGain,
                              verbose=verbose)
    return [x[0] for x in greedyList]


def subset(lake_data,
           query_data,
           eta,
           image_list,
           strategry,
           private_data=None,
           budget=10,
           metric="cosine",
           optimizer="NaiveGreedy",
           stopIfZeroGain=False,
           stopIfNegativeGain=False,
           verbose=False,
           kernel=False):

    if (kernel):
        if (strategry == "fl2mi"):
            subset = FL2MI_wrapper_kernel(query_data,
                                          eta,
                                          budget=budget,
                                          metric="cosine",
                                          optimizer="LazyGreedy",
                                          stopIfZeroGain=False,
                                          stopIfNegativeGain=False,
                                          verbose=False)
        elif (strategry == "gcmi"):
            subset = GCMI_wrapper_kernel(query_data,
                                         eta,
                                         budget=budget,
                                         metric="cosine",
                                         optimizer="LazyGreedy",
                                         stopIfZeroGain=False,
                                         stopIfNegativeGain=False,
                                         verbose=False)
        elif (strategry == "com"):
            subset = COM_wrapper_kernel(query_data,
                                        eta,
                                        budget=budget,
                                        metric="cosine",
                                        optimizer="LazyGreedy",
                                        stopIfZeroGain=False,
                                        stopIfNegativeGain=False,
                                        verbose=False)
    else:

        if (len(lake_data) < budget):
            rem_budget = budget - len(lake_data) + 1
            budget = len(lake_data) - 1

        rem_budget = 0
        if (strategry == "fl1mi"):
            subset = FL1MI_wrapper(lake_data,
                                   query_data,
                                   1.5,
                                   image_list,
                                   budget=budget,
                                   metric=metric,
                                   optimizer="LazyGreedy",
                                   stopIfZeroGain=False,
                                   stopIfNegativeGain=False,
                                   verbose=False)
        elif (strategry == "fl2mi"):
            subset = FL2MI_wrapper(lake_data,
                                   query_data,
                                   eta,
                                   image_list,
                                   budget=budget,
                                   metric="cosine",
                                   optimizer="LazyGreedy",
                                   stopIfZeroGain=False,
                                   stopIfNegativeGain=False,
                                   verbose=False)
        elif (strategry == "com"):
            subset = COM_wrapper(lake_data,
                                 query_data,
                                 eta,
                                 image_list,
                                 budget=budget,
                                 metric="cosine",
                                 optimizer="LazyGreedy",
                                 stopIfZeroGain=False,
                                 stopIfNegativeGain=False,
                                 verbose=False)
        elif (strategry == "gcmi"):
            subset = GCMI_wrapper(lake_data,
                                  query_data,
                                  eta,
                                  image_list,
                                  budget=budget,
                                  metric="cosine",
                                  optimizer="LazyGreedy",
                                  stopIfZeroGain=False,
                                  stopIfNegativeGain=False,
                                  verbose=False)
        elif (strategry == "logdet"):
            subset = LogDetMI_wrapper(lake_data,
                                      query_data,
                                      1,
                                      image_list,
                                      budget=budget,
                                      metric="cosine",
                                      optimizer="LazyGreedy",
                                      stopIfZeroGain=False,
                                      stopIfNegativeGain=False,
                                      verbose=False)
        elif (strategry == 'cmi'):
            subset = FL1CMI_wrapper(lake_data,
                                    query_data,
                                    private_data,
                                    eta,
                                    image_list,
                                    budget=budget,
                                    metric="cosine",
                                    optimizer="LazyGreedy",
                                    stopIfZeroGain=False,
                                    stopIfNegativeGain=False,
                                    verbose=False)

        if (rem_budget > 0):
            all_lake_idx = list(range(len(image_list)))
            remain_lake_idx = list(set(all_lake_idx) - set(subset))
            random_subset_idx = list(
                np.random.choice(np.array(remain_lake_idx),
                                 size=int(rem_budget),
                                 replace=False))
            subset += random_subset_idx

    return [image_list[i] for i in subset]


def submod_results(lake_data,
                   query_data,
                   eta,
                   targeted_classes,
                   image_list,
                   annotations,
                   image_areas,
                   category_list,
                   budget=10,
                   metric='cosine'):
    Random_results = []
    FL1MI_results = []
    FL2MI_results = []
    COM_results = []
    GCMI_results = []
    LogDetMI_results = []

    # Reandom sampling
    Random_results = Random_wrapper(image_list, budget)

    # FL1MI
    FL1MI_results = FL1MI_wrapper(lake_data,
                                  query_data,
                                  eta,
                                  image_list,
                                  budget=budget,
                                  metric=metric,
                                  optimizer="NaiveGreedy",
                                  stopIfZeroGain=False,
                                  stopIfNegativeGain=False,
                                  verbose=False)

    # FL2MI
    FL2MI_results = FL2MI_wrapper(lake_data,
                                  query_data,
                                  eta,
                                  image_list,
                                  budget=10,
                                  metric="cosine",
                                  optimizer="NaiveGreedy",
                                  stopIfZeroGain=False,
                                  stopIfNegativeGain=False,
                                  verbose=False)

    # COM
    COM_results = COM_wrapper(lake_data,
                              query_data,
                              eta,
                              image_list,
                              budget=10,
                              metric="cosine",
                              optimizer="LazierThanLazyGreedy",
                              stopIfZeroGain=False,
                              stopIfNegativeGain=False,
                              verbose=False)

    # GCMI
    GCMI_results = GCMI_wrapper(lake_data,
                                query_data,
                                eta,
                                image_list,
                                budget=10,
                                metric="cosine",
                                optimizer="LazierThanLazyGreedy",
                                stopIfZeroGain=False,
                                stopIfNegativeGain=False,
                                verbose=False)

    # LogDetMI
    if eta < 3:
        LogDetMI_results = LogDetMI_wrapper(lake_data,
                                            query_data,
                                            eta,
                                            image_list,
                                            budget=10,
                                            metric="cosine",
                                            optimizer="LazierThanLazyGreedy",
                                            stopIfZeroGain=False,
                                            stopIfNegativeGain=False,
                                            verbose=False,
                                            lambdaVal=1)

    target_ids = [category_list.index(target) for target in targeted_classes]

    Random_percent = []
    for f in Random_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        Random_percent.append(percent_area_covered)

    FL1MI_percent = []
    for f in FL1MI_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        FL1MI_percent.append(percent_area_covered)

    FL2MI_percent = []
    for f in FL2MI_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        FL2MI_percent.append(percent_area_covered)

    COM_percent = []
    for f in COM_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        COM_percent.append(percent_area_covered)

    GCMI_percent = []
    for f in GCMI_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        GCMI_percent.append(percent_area_covered)

    LogDetMI_percent = []
    for f in LogDetMI_results:
        area_covered = 0
        annotation = annotations[f]
        for ann in annotation:
            if ann[1] in target_ids:
                area_covered += ann[0]

        percent_area_covered = area_covered / image_areas[f]
        LogDetMI_percent.append(percent_area_covered)

    return Random_percent, FL1MI_percent, FL2MI_percent, COM_percent, GCMI_percent, LogDetMI_percent
