from predictor_model.processing import preprocessing, text_pipeline, dataloaders


def pipeline(data, train=True):
    data = preprocessing.preprocess_salary(data)
    if train:
        data = preprocessing.preprocess_description(data)
        # maybe only save vocab if doesn't exist??
        data = text_pipeline.create_and_save_vocab(data)
    data = text_pipeline.text_pipeline(data)
    dataloader = dataloaders.create_dataloader(data)
    return dataloader
