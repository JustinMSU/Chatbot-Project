<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <link href="css/style.css" rel="stylesheet">
    <title>Chatbot Main</title>
</head>
<body class="main">
    <h1 class="floating-title">Talk with a Chatbot</h1>
    <div class="content">
        <div class="container-left">
            <h1 class="title">Chat Here</h1>
            <form class="input_form" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?>" method="post">
                <input class="form-input" type="text" id="inputtxt" name="inputtxt" placeholder="Converse...">
                <button class="form-button" type="submit" id="enterB" name="enterB">Enter</button>
            </form>

        </div>
        <div class="container-right">
            <div class="conv-content">
                <span>Past conversation here</span>
            </div>
        </div>
    </div>
    
</body>
</html>
