' Local Image Studio - hidden launcher for autorun.bat.
' Started by the Startup shortcut at every sign-in
' (created by install-autostart.bat). No window ever appears.
Option Explicit
Dim sh, fso, dir
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = dir
' window style 0 = hidden, False = do not wait (autorun loops forever)
sh.Run "cmd /c """ & dir & "\autorun.bat""", 0, False
