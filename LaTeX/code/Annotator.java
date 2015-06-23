/**
 * Annotator.java        made by: Ysbrand Galama, februari 2015
 *
 * This program can be used to easaly annotate large amounts
 * of image-datasets.
 * 
 * call using: java Annotator [name of output-file] [path to image directory]
 * 
 * controls:
 * UP and DOWN arrows    :  go to next and previous image
 * LEFT and RIGHT arrows :  go to next and previous image, while copying their annotation to the new image
 * a,s,d,f keys          :  toggle annotation of the class in the current image
 */

import java.io.*;
import java.nio.file.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import javax.swing.*;
import javax.imageio.*;
import java.lang.*;
import java.util.Date;
import java.text.*;
import javax.swing.text.*;
import java.lang.Class;
import java.util.Calendar;

@SuppressWarnings("serial")
public class Annotator extends JFrame{

	private ImagePanel panel;
	private BufferedImage image;
	private String name = "DATA001", path = "../"+name+".wmvout/", ext = ".jpg";
	private int index = 1;
	private int MAX = new File(path).list().length;
	private File file;
	private boolean[][] lines = new boolean[MAX+1][4];
	//private JTextField textfield = new JTextField();

	public Annotator(String name, String path) {
		super("Annotator");
		this.name = name;
		this.path = path;
		open();
		panel = new ImagePanel();
		panel.setPreferredSize(new Dimension(960,538));
		panel.getInputMap().put(KeyStroke.getKeyStroke("F"), "next");
		panel.getInputMap().put(KeyStroke.getKeyStroke("S"), "next");
		panel.getInputMap().put(KeyStroke.getKeyStroke("A"), "next");
		panel.getInputMap().put(KeyStroke.getKeyStroke("D"), "next");
		panel.getInputMap().put(KeyStroke.getKeyStroke("UP"), "up");
		panel.getInputMap().put(KeyStroke.getKeyStroke("DOWN"), "down");
		panel.getInputMap().put(KeyStroke.getKeyStroke("LEFT"), "left");
		panel.getInputMap().put(KeyStroke.getKeyStroke("RIGHT"), "right");
		panel.getActionMap().put("next", new AnnotatorAction());
		panel.getActionMap().put("up", new AnnotatorAction("UP"));
		panel.getActionMap().put("down", new AnnotatorAction("DOWN"));
		panel.getActionMap().put("left", new AnnotatorAction("LEFT"));
		panel.getActionMap().put("right", new AnnotatorAction("RIGHT"));
		
		JPanel overall = new JPanel(new BorderLayout());
		JPanel top = new JPanel(new GridLayout(0,4,5,5));
		
		top.add(new PlasticPanel(2,"Animals"));
		top.add(new PlasticPanel(0,"BelowWater"));
		top.add(new PlasticPanel(1,"Plastic"));
		top.add(new PlasticPanel(3,"Suited"));
		
		overall.add(panel,BorderLayout.CENTER);
		overall.add(top,BorderLayout.PAGE_START);
		
		getContentPane().add(overall);
		
		setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
		this.addWindowListener(new WindowAdapter(){
			@Override
			public void windowClosing(WindowEvent e){
				if (save()) {
					System.exit(0);
				}
			}
		});
		
		pack();
		setLocationRelativeTo(null);
		setResizable(true);
		setVisible(true);
	}
	
	/**
	 * open() reads an output file if it already exists
	 */
	private void open() {
		try{
			BufferedReader bf = new BufferedReader(new FileReader(new File("../"+name+".txt")));
			for (int i=0; i < MAX; i++) {
				String line = bf.readLine();
				lines[i][0] = (line.charAt(0) == '1');
				lines[i][1] = (line.charAt(1) == '1');
				lines[i][2] = (line.charAt(2) == '1');
				lines[i][3] = (line.charAt(3) == '1');
			}
			System.out.println("read: "+"../"+name+".txt");
		} catch (Exception e) {
			System.out.println("No file read");
		}
	}
	
	/**
	 * save() overwrites the current output file if it exists with 
	 * the changes made in the data
	 */
	public boolean save() {
		System.out.println("SAVING");
		try{
			PrintWriter writer = new PrintWriter("../"+name+".txt", "UTF-8");
			for (int i=0; i < MAX; i++) {
				writer.printf("%d%d%d%d\n",
						(lines[i][0] ? 1 : 0),
						(lines[i][1] ? 1 : 0),
						(lines[i][2] ? 1 : 0),
						(lines[i][3] ? 1 : 0) );
			}
			writer.close();
			return true;
		} catch (Exception e) {
			System.out.println(e);
			return false;
		}
	}
	
	/**
	 * read() opens and shows the next image file
	 */
	public void read() {
		try {                
			image = ImageIO.read(file);
			System.out.println("opened: "+file);
			//textfield.setText(file.toString());
			repaint();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
	
	/**
	 * nextFillImage() copies the annotation of the previous image
	 * and opens the next one
	 */
	private void nextFillImage() {
		boolean a = lines[index][0], 
				b = lines[index][1],
				c = lines[index][2],
				d = lines[index][3];
		if (index++ >= MAX) {
			index = 1;
		}
		lines[index][0] = a;
		lines[index][1] = b;
		lines[index][2] = c;
		lines[index][3] = d;
		System.out.print("autofill ");
		
		file = new File(path+index+ext);
		read();
	}
	
	/**
	 * nextImage() opens the next image
	 */
	private void nextImage() {
		if (index++ >= MAX) {
			index = 1;
		}
		file = new File(path+index+ext);
		read();
	}
	
	/**
	 * previousImage() is the opposide of nextImage()
	 */
	private void previousImage() {
		if (index-- <= 1) {
			index = MAX;
		}
		file = new File(path+index+ext);
		read();
	}
	
	/**
	 * previousFillImage() is the opposide of nextFillImage()
	 */
	private void previousFillImage() {
		boolean a = lines[index][0], 
				b = lines[index][1],
				c = lines[index][2],
				d = lines[index][3];
		if (index-- <= 1) {
			index = MAX;
		}
		lines[index][0] = a;
		lines[index][1] = b;
		lines[index][2] = c;
		lines[index][3] = d;
		System.out.print("autofill ");
		
		file = new File(path+index+ext);
		read();
	}
	
	/**
	 * setX() changes the annotation of the class
	 */
	private void setBoven() {
		lines[index][0] = !lines[index][0];
		repaint();
	}
	private void setPlastic() {
		lines[index][1] = !lines[index][1];
		repaint();
	}
	private void setAnimal() {
		lines[index][2] = !lines[index][2];
		repaint();
	}
	private void setSave() {
		lines[index][3] = !lines[index][3];
		repaint();
	}
	
	/**
	 * PlasticPanel is a sub-class that draws the annotations on screen
	 */
	private class PlasticPanel extends JPanel {
		private int i;
		private String name;
		
		public PlasticPanel(int i, String name) {
			super();
			setMinimumSize(new Dimension(50,30));
			setPreferredSize(new Dimension(50,30));
			this.i = i;
			this.name = name;
		}
		@Override
		protected void paintComponent(Graphics g) {
			Rectangle r = this.getBounds();
			g.setColor(Color.RED);
			if (lines[index][i] ) 
				g.setColor(Color.GREEN);
			g.fillRect(0, 0, r.width, r.height);
			g.setColor(Color.BLACK);
			int strLen = (int) g.getFontMetrics().getStringBounds(name, g).getWidth();
			g.drawString(name,r.width/2-strLen/2,r.height/2);
		}
	}
	
	/**
	 * ImagePanel is a sub-class that shows the image
	 */
	private class ImagePanel extends JPanel {
		@Override
		protected void paintComponent(Graphics g) {
			Rectangle r = this.getBounds();
			g.drawImage(image, r.x, r.y, null);
		}
	}
	
	/**
	 * AnnotatorAction is a sub-class that handles the key-strokes of the user
	 */
	private class AnnotatorAction extends AbstractAction {
		public AnnotatorAction() {
			super();
		}
		public AnnotatorAction(String text) {
			putValue(ACTION_COMMAND_KEY, text);
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			String c = (String) e.getActionCommand();
			//System.out.println("ac "+c);
			if (c == null) {
				return;
			} else if (c.equals("UP")) {
				previousImage();
			} else if (c.equals("DOWN")) {
				nextImage();
			} else if (c.equals("RIGHT")) {
				nextFillImage();
			} else if (c.equals("LEFT")) {
				previousFillImage();
			} else if (c.equals("s")) {
				setBoven();
			} else if (c.equals("d")) {
				setPlastic();
			} else if (c.equals("a")) {
				setAnimal();
			} else if (c.equals("f")) {
				setSave();
			}
		}
	}

	public static void main(String[] args) {
		try{
			String name = args[0];
			String path = args[1];
			if (!(new File(path)).isDirectory()){
				throw new Exception();
			}
			new Annotator(name,path);
		} catch (Exception e) {
			System.out.println("use as follows: java Annotator [name of output-file] [path to images-directory]");
		}
	}
}