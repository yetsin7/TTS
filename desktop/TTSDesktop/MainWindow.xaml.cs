using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using TTSDesktop.Services;

namespace TTSDesktop;

/// <summary>
/// Ventana principal de la app WPF para TTS.
/// </summary>
public partial class MainWindow : Window
{
    private readonly PythonBridge _pythonBridge;
    private string? _latestAudioPath;

    public MainWindow()
    {
        InitializeComponent();
        _pythonBridge = new PythonBridge();
        PathsTextBlock.Text = string.Join(
            Environment.NewLine,
            [
                $"- Audio recomendado: {Path.Combine(_pythonBridge.RootDir, "data", "raw", "audio_entrenamiento.wav")}",
                $"- Audio alterno: {Path.Combine(_pythonBridge.RootDir, "data", "raw", "audio_entrenamiento.mp3")}",
                $"- Audio alterno: {Path.Combine(_pythonBridge.RootDir, "data", "raw", "audio_entrenamiento.flac")}",
                $"- Texto requerido: {Path.Combine(_pythonBridge.RootDir, "data", "raw", "texto_entrenamiento.txt")}",
            ]
        );
        AppendLog("La interfaz WPF esta lista.");
    }

    private async void OnPreflightClick(object sender, RoutedEventArgs e) => await RunAndLogAsync("scripts/preflight.py");
    private async void OnPrepareClick(object sender, RoutedEventArgs e) => await RunAndLogAsync("scripts/prepare_dataset.py");
    private async void OnTrainClick(object sender, RoutedEventArgs e) => await RunAndLogAsync("scripts/train.py");
    private async void OnPipelineClick(object sender, RoutedEventArgs e) => await RunAndLogAsync("scripts/run_pipeline.py");

    private async void OnSynthesizeClick(object sender, RoutedEventArgs e)
    {
        var prompt = PromptTextBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(prompt))
        {
            AppendError("Escribe un texto antes de generar audio.");
            return;
        }

        var escaped = prompt.Replace("\"", "\\\"");
        var result = await RunAndLogAsync("scripts/synthesize.py", $"--text \"{escaped}\"", "Sintetizando audio...");
        if (!result.Success)
        {
            return;
        }

        _latestAudioPath = Path.Combine(_pythonBridge.RootDir, "outputs", "generated", "tts_output.wav");
        LatestAudioTextBlock.Text = _latestAudioPath;
        HistoryTextBox.AppendText($"TEXTO: {prompt}{Environment.NewLine}AUDIO: {_latestAudioPath}{Environment.NewLine}{Environment.NewLine}");
        HistoryTextBox.ScrollToEnd();
    }

    private void OnPlayClick(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_latestAudioPath) || !File.Exists(_latestAudioPath))
        {
            AppendError("Todavia no existe un audio generado para reproducir.");
            return;
        }

        using var player = new System.Media.SoundPlayer(_latestAudioPath);
        player.Play();
        AppendLog($"Reproduciendo {_latestAudioPath}");
    }

    private void OnOpenOutputClick(object sender, RoutedEventArgs e)
    {
        var outputDir = Path.Combine(_pythonBridge.RootDir, "outputs", "generated");
        Directory.CreateDirectory(outputDir);
        Process.Start(new ProcessStartInfo
        {
            FileName = "explorer.exe",
            Arguments = outputDir,
            UseShellExecute = true,
        });
    }

    private async Task<ProcessResult> RunAndLogAsync(string scriptPath, string? args = null, string? initialMessage = null)
    {
        ToggleButtons(false);
        if (!string.IsNullOrWhiteSpace(initialMessage))
        {
            AppendLog(initialMessage);
        }

        AppendLog($">>> Ejecutando {scriptPath} {args}".Trim());
        try
        {
            var result = await _pythonBridge.RunScriptAsync(scriptPath, args);
            if (result.Success)
            {
                AppendLog(result.Output);
            }
            else
            {
                AppendError(result.Output);
            }
            return result;
        }
        catch (Exception exception)
        {
            AppendError(exception.ToString());
            return new ProcessResult(false, exception.Message);
        }
        finally
        {
            ToggleButtons(true);
        }
    }

    private void ToggleButtons(bool enabled)
    {
        foreach (var button in FindVisualChildren<Button>(this))
        {
            button.IsEnabled = enabled;
        }
    }

    private void AppendLog(string message)
    {
        LogsTextBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}{Environment.NewLine}");
        LogsTextBox.ScrollToEnd();
    }

    private void AppendError(string message)
    {
        AppendLog($"[ERROR] {message}");
        MessageBox.Show(this, message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
    }

    private static IEnumerable<T> FindVisualChildren<T>(DependencyObject parent) where T : DependencyObject
    {
        for (var i = 0; i < System.Windows.Media.VisualTreeHelper.GetChildrenCount(parent); i++)
        {
            var child = System.Windows.Media.VisualTreeHelper.GetChild(parent, i);
            if (child is T typedChild)
            {
                yield return typedChild;
            }

            foreach (var descendant in FindVisualChildren<T>(child))
            {
                yield return descendant;
            }
        }
    }
}
