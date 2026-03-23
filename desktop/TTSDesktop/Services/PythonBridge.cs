using System.Diagnostics;
using System.IO;
using System.Text;

namespace TTSDesktop.Services;

/// <summary>
/// Ejecuta scripts Python del proyecto y devuelve su salida.
/// </summary>
public sealed class PythonBridge
{
    private readonly string _rootDir;
    private readonly string _pythonExe;

    public PythonBridge()
    {
        _rootDir = FindRootDir();
        _pythonExe = Path.Combine(_rootDir, ".venv", "Scripts", "python.exe");
        if (!File.Exists(_pythonExe))
        {
            throw new FileNotFoundException($"No existe el entorno virtual esperado en {_pythonExe}");
        }
    }

    public string RootDir => _rootDir;

    public async Task<ProcessResult> RunScriptAsync(string relativeScript, string? arguments = null)
    {
        var scriptPath = Path.Combine(_rootDir, relativeScript);
        if (!File.Exists(scriptPath))
        {
            return new ProcessResult(false, $"No existe el script: {scriptPath}");
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExe,
            WorkingDirectory = _rootDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            UseShellExecute = false,
            CreateNoWindow = true,
            Arguments = string.IsNullOrWhiteSpace(arguments)
                ? $"\"{scriptPath}\""
                : $"\"{scriptPath}\" {arguments}",
        };

        using var process = new Process { StartInfo = startInfo };
        var output = new StringBuilder();
        process.Start();

        var stdoutTask = Task.Run(async () =>
        {
            while (!process.StandardOutput.EndOfStream)
            {
                output.AppendLine(await process.StandardOutput.ReadLineAsync());
            }
        });

        var stderrTask = Task.Run(async () =>
        {
            while (!process.StandardError.EndOfStream)
            {
                output.AppendLine(await process.StandardError.ReadLineAsync());
            }
        });

        await Task.WhenAll(stdoutTask, stderrTask, process.WaitForExitAsync());
        return new ProcessResult(process.ExitCode == 0, output.ToString().Trim());
    }

    private static string FindRootDir()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "pyproject.toml")))
            {
                return current.FullName;
            }
            current = current.Parent;
        }

        throw new DirectoryNotFoundException("No se encontro la raiz del proyecto TTS.");
    }
}

/// <summary>
/// Resultado simple de un proceso Python.
/// </summary>
public sealed record ProcessResult(bool Success, string Output);
