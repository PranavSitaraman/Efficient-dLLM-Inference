

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{applemlr}
\LoadClassWithOptions{article}



\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage[expansion=false]{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}

\DisableLigatures[f]{family=sf*}

\RequirePackage{graphicx}
\RequirePackage{subcaption}


\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}


\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\definecolor{bscolor}{HTML}{007AFF}
\definecolor{fgcolor}{HTML}{333333}
\definecolor{bgcolor}{HTML}{EEEEEE}


\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=bscolor,
  citecolor=bscolor,
  urlcolor=bscolor
}


\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}

% ===== FONT SETUP =====
\RequirePackage[T1]{fontenc}

% For pdflatex: create proper SF Pro font installation using otftotfm-generated files
\DeclareFontFamily{T1}{sfpro}{}

% Regular weight, upright shape
\DeclareFontShape{T1}{sfpro}{m}{n}{%
  <-> s*[0.88] store/SFPro-Regular%
}{}

% Regular weight, italic shape
\DeclareFontShape{T1}{sfpro}{m}{it}{%
  <-> s*[0.88] store/SFPro-Italic%
}{}

% Regular weight, slanted shape (substitute italic)
\DeclareFontShape{T1}{sfpro}{m}{sl}{%
  <-> s*[0.88] store/SFPro-Italic%
}{}

% Bold weight, upright shape
\DeclareFontShape{T1}{sfpro}{b}{n}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Bold weight, italic shape
\DeclareFontShape{T1}{sfpro}{b}{it}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Bold weight, slanted shape
\DeclareFontShape{T1}{sfpro}{b}{sl}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Bold extended weight (substitute bold)
\DeclareFontShape{T1}{sfpro}{bx}{n}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Bold extended weight, italic
\DeclareFontShape{T1}{sfpro}{bx}{it}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Bold extended weight, slanted
\DeclareFontShape{T1}{sfpro}{bx}{sl}{%
  <-> s*[0.88] store/SFPro-Bold%
}{}

% Set SF Pro as default sans serif
\renewcommand{\sfdefault}{sfpro}

% Load font maps at begin document to ensure proper timing
\AtBeginDocument{%
  \ifPDFTeX
    % Add individual font map lines
    \pdfmapline{+SFPro-Regular SFPro-Regular "AutoEnc_bimro2hc43zdf4rmkcldbhoftd ReEncodeFont" <[store/a_bimro2.enc <store/SF-Pro.ttf}%
    \pdfmapline{+SFPro-Italic SFPro-RegularItalic "AutoEnc_kio5jvs3t76a3hixgi6okbojxe ReEncodeFont" <[store/a_kio5jv.enc <store/SF-Pro-Italic.ttf}%
    \pdfmapline{+SFPro-Bold SFProDisplay-Bold "AutoEnc_oua5llbf4mn2stqy7jkt2gtlib ReEncodeFont" <[store/a_oua5ll.enc <store/SF-Pro-Display-Bold.ttf}%
  \fi
}

% Create helper commands
\DeclareRobustCommand{\sbseries}{\fontseries{sb}\selectfont}
\DeclareTextFontCommand{\textsb}{\sbseries}

\RequirePackage{titlesec}

\titleformat*{\paragraph}{\sffamily\bfseries}
\titleformat*{\section}{\Large\sffamily\bfseries}
\titleformat*{\subsection}{\large\sffamily\bfseries}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries}
\DeclareTextFontCommand{\sftextbf}{\bfseries\sffamily}




\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\sftextbf{#1 #2}} #3}
\captionsetup{singlelinecheck=true,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}


\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{\sffamily, }}



\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize \sffamily $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{\sffamily, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\footnotesize $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{\sffamily, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily \bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\renewcommand{\abstract}[1]{\def\abstractlist{{\sffamily \color{fgcolor} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\renewcommand{\title}[1]{\newcommand{\titlelist}{{\Huge\sffamily\bfseries #1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}



\newcommand{\mymaketitle}{%
  \tcbset{enhanced,frame hidden}
  \tcbset{left=0.5cm}
  \tcbset{right=0.5cm}
  \tcbset{top=0.5cm}
  \tcbset{bottom=0.5cm}
  \tcbset{arc=16pt}
  \tcbset{before skip=0pt}
  \tcbset{grow to left by=1.5pt}
  \tcbset{grow to right by=1.5pt}
  \tcbset{overlay={\node[
    anchor=north west,
    at= (frame.north west),
    xshift= 0.5cm,
    yshift= -0.5cm] {\includegraphics[width=0.5cm]{store/apple_logo.pdf}};}}

  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.5cm}
    {
      \setlength{\parskip}{0cm}
      \raggedright
      \nohyphens
      %
      {
        \vskip 1.5cm
        \begingroup
          \setlength{\baselineskip}{2.0\baselineskip}
          \setlength{\lineskiplimit}{-\maxdimen}
          \titlelist\par
        \endgroup
      }
      \vskip 0.2cm
      %
      \authorlist\par
      \vskip 0.2cm
      %
      \affiliationlist\par
      %
      \contributionlist\par
    }
    %
    \abstractlist\par
    %
    \vskip 0.5cm
    %
    {
      \setlength{\parskip}{0cm}
      \ifdefempty{\metadatalist}{\vspace*{0.65cm}}{\metadatalist\par}
    }

  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.38cm
]
}%
\else%
\renewcommand{\maketitle}{
  \mymaketitle
}%
\fi
