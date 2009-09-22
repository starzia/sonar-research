' this scripts clears out the sonarPM config/logging directory

Set WshShell = WScript.CreateObject("Wscript.Shell")
vAPPDATA = WshShell.ExpandEnvironmentStrings("%APPDATA%")
Set objFSO = CreateObject("Scripting.FileSystemObject")
objFSO.DeleteFile(vAPPDATA & "\sonarPM\*")
'objFSO.DeleteFolder(vAPPDATA & "\FolderName")