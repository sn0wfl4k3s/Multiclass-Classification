using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace ConsoleApp2
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Dados", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Dados", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Modelos", "model.zip");

        static void Main(string[] args)
        {
            var _mlContext = new MLContext(seed: 0);

            //var _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            //var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
            //    .Append(_mlContext.Transforms.Text.FeaturizeText(
            //        inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
            //    .Append(_mlContext.Transforms.Text.FeaturizeText(
            //        inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
            //    .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
            //    .AppendCacheCheckpoint(_mlContext);

            //var trainingPipeline = pipeline
            //    .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            //    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //var _trainedModel = trainingPipeline.Fit(_trainingDataView);

            //_mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, _modelPath);

            var _trainedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            var issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"Prediction: {prediction.Area}");
        }
    }
}
